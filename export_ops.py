import os
import json
import bpy
from mathutils import Matrix, Vector


from .utils import (
    ensure_dir, timestamp,
    mat4_to_list, mat3_to_list, vec3_to_list,
    compute_camera_K, get_extrinsics, compute_projection_matrix_gl,
    compute_viewport_calibration,     # ← 新增
)

from .utils import create_blank_png

# ---------- 渲染：相机 ----------
def render_from_camera(scene, cam_obj, out_dir, basename):
    """正式渲染（使用选定相机）为 PNG 文件，返回路径。"""
    ensure_dir(out_dir)

    # 记录现场
    orig_cam = scene.camera
    orig_filepath = scene.render.filepath
    img_settings = scene.render.image_settings
    orig_format = img_settings.file_format

    out_path_noext = os.path.join(out_dir, f"{basename}_camera")
    try:
        scene.camera = cam_obj
        scene.render.filepath = out_path_noext
        img_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

        out_path = scene.render.filepath
        if not out_path.lower().endswith(".png"):
            out_path += ".png"
    finally:
        scene.camera = orig_cam
        scene.render.filepath = orig_filepath
        img_settings.file_format = orig_format

    return out_path

# ---------- 渲染：视口 ----------
def render_from_viewport(context, out_dir, basename, vp_w, vp_h):
    """视口渲染（Viewport Render）为 PNG 文件，返回路径。"""
    ensure_dir(out_dir)
    scene = context.scene

    # 定位第一个 3D 视口
    win = None; area = None; region = None; space = None
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type != 'VIEW_3D':
                continue
            reg_win = None
            for r in a.regions:
                if r.type == 'WINDOW':
                    reg_win = r
                    break
            if not reg_win:
                continue
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    win, area, region, space = w, a, reg_win, s
                    break
            if win and area and region and space:
                break
        if win and area and region and space:
            break

    if not (win and area and region and space):
        raise RuntimeError("未找到 3D 视口窗口，请确保至少打开一个 3D Viewport。")

    # 记录现场
    orig_filepath = scene.render.filepath
    img_settings = scene.render.image_settings
    orig_format = img_settings.file_format
    orig_rx = scene.render.resolution_x
    orig_ry = scene.render.resolution_y
    orig_rp = scene.render.resolution_percentage

    out_path_noext = os.path.join(out_dir, f"{basename}_viewport")

    try:
        scene.render.filepath = out_path_noext
        img_settings.file_format = 'PNG'
        # 设定视口输出尺寸
        scene.render.resolution_x = int(vp_w)
        scene.render.resolution_y = int(vp_h)
        scene.render.resolution_percentage = 100

        with context.temp_override(window=win, area=area, region=region, space_data=space, scene=scene):
            bpy.ops.render.opengl(write_still=True, view_context=True)

        out_path = scene.render.filepath
        if not out_path.lower().endswith(".png"):
            out_path += ".png"

    finally:
        # 恢复现场
        scene.render.filepath = orig_filepath
        img_settings.file_format = orig_format
        scene.render.resolution_x = orig_rx
        scene.render.resolution_y = orig_ry
        scene.render.resolution_percentage = orig_rp

    return out_path


# ---------- UV 导出 ----------
def collect_object_uv_data(obj):
    me = obj.data
    uv_layer = me.uv_layers.active
    data = {
        "uv_map": uv_layer.name if uv_layer else None,
        "uvs": [],
        "loops_count": len(me.loops),
        "polygons": len(me.polygons),
        "vertices": len(me.vertices),
    }
    if uv_layer:
        uvs = []
        for luv in uv_layer.data:
            uvs.append([float(luv.uv.x), float(luv.uv.y)])
        data["uvs"] = uvs
    return data

def export_uv_layout_png(obj, out_dir, basename, size=4096):
    """导出活动 UV 的轮廓图（PNG）。"""
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{basename}_uv.png")

    prev_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
    prev_active = bpy.context.view_layer.objects.active

    bpy.context.view_layer.objects.active = obj
    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.uv.export_layout(
        filepath=path,
        export_all=True,
        modified=False,
        mode='PNG',
        size=(size, size),
        opacity=0.25,
    )

    bpy.ops.object.mode_set(mode=prev_mode)
    bpy.context.view_layer.objects.active = prev_active
    return path



class PSB_OT_Export(bpy.types.Operator):
    bl_idname = "psb.export_camera_uv"
    bl_label = "输出（渲染+相机参数+UV）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.psb_exporter
        mode_render = props.render_mode  # "CAMERA" or "VIEWPORT"
        out_dir = bpy.path.abspath(props.out_dir)
        uv_size = int(props.uv_size)

        # -------- 收集目标 --------
        targets = []
        if getattr(props, "bake_target_mode", "SINGLE") == "COLLECTION":
            col = getattr(props, "target_collection", None)
            if col:
                targets = [o for o in col.objects if getattr(o, "type", "") == 'MESH']
        else:
            obj = getattr(props, "target_object", None)
            if obj and getattr(obj, "type", "") == 'MESH':
                targets = [obj]

        if not targets:
            self.report({'ERROR'}, "未找到要导出的网格物体（请检查目标类型与目标选择）")
            return {'CANCELLED'}

        # -------- 相机/视口前置检查 --------
        cam = props.camera
        if mode_render == "CAMERA" and (not cam or cam.type != 'CAMERA'):
            self.report({'ERROR'}, "渲染源为相机，但未选择有效相机")
            return {'CANCELLED'}

        ensure_dir(out_dir)

        # -------- 命名基准 --------
        # 相机名：相机模式用 cam.name；视口模式用 "Viewport"
        camera_name = cam.name if (mode_render == "CAMERA" and cam) else "Viewport"

        # Blender 文件名（用于 metadata.json）
        blend_path = bpy.data.filepath
        blend_base = os.path.splitext(os.path.basename(blend_path))[0] if blend_path else "untitled"

        # -------- 渲染一次：相机名_camera.png --------
        try:
            if mode_render == "CAMERA":
                # render_from_camera 会生成 <basename>_camera.png，因此 basename 直接用 camera_name
                image_path_tmp = render_from_camera(context.scene, cam, out_dir, camera_name)
                image_path = image_path_tmp  # 已经是 .../<相机名>_camera.png
            else:
                # 视口函数会生成 <basename>_viewport.png；我们强制改名成 <相机名>_camera.png
                image_path_tmp = render_from_viewport(
                    context, out_dir, camera_name, int(props.viewport_width), int(props.viewport_height)
                )
                image_path = os.path.join(out_dir, f"{camera_name}_camera.png")
                try:
                    if os.path.abspath(image_path_tmp) != os.path.abspath(image_path):
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        os.replace(image_path_tmp, image_path)
                except Exception:
                    # 如果改名失败，就用原始路径，但后续仍按 image_path 记录
                    image_path = image_path_tmp
        except Exception as e:
            self.report({'ERROR'}, f"渲染失败：{e}")
            return {'CANCELLED'}

        # -------- 计算 3D 画布尺寸并生成：相机名_paint_3d.png --------
        if mode_render == "VIEWPORT":
            rw, rh = int(props.viewport_width), int(props.viewport_height)
        else:
            s = context.scene
            rw = int(s.render.resolution_x * (s.render.resolution_percentage / 100.0))
            rh = int(s.render.resolution_y * (s.render.resolution_percentage / 100.0))

        paint3d_path = os.path.join(out_dir, f"{camera_name}_paint_3d.png")
        # 使用烘焙背景色作为初始颜色
        bg = tuple(getattr(props, "bake_background_color", (1.0, 1.0, 1.0, 0.0)))
        create_blank_png(rw, rh, paint3d_path, color=bg)

        # -------- 相机参数（若有） --------
        K = cam_params = cam_to_world = world_to_cam = R = t = proj = None
        if cam and cam.type == 'CAMERA':
            try:
                K, cam_params = compute_camera_K(context.scene, cam)
                cam_to_world, world_to_cam, R, t = get_extrinsics(cam)
                proj = compute_projection_matrix_gl(context.scene, cam)
            except Exception as e:
                self.report({'WARNING'}, f"相机参数计算失败：{e}")

        # -------- 视口标定（尽量写） --------
        viewport_calib = None
        try:
            if mode_render == "VIEWPORT":
                vp_w = int(props.viewport_width)
                vp_h = int(props.viewport_height)
            else:
                s = context.scene
                vp_w = int(s.render.resolution_x * (s.render.resolution_percentage / 100.0))
                vp_h = int(s.render.resolution_y * (s.render.resolution_percentage / 100.0))
            viewport_calib = compute_viewport_calibration(context, vp_w, vp_h)
        except Exception:
            viewport_calib = None

        # -------- 为每个物体导出：UV 示意图（线框轮廓）到 out_dir/uv/ --------
        per_object_entries = []
        uv_dir = os.path.join(out_dir, "uv")
        ensure_dir(uv_dir)
        for obj in targets:
            try:
                # 导出活动 UV 的轮廓图（示意图）
                uv_png_path = export_uv_layout_png(obj, uv_dir, obj.name, size=uv_size)

                # UV 数据（用于写入 metadata；如不需要可删掉 uv_data 字段）
                uv_data = collect_object_uv_data(obj)

                # 物体信息
                from mathutils import Vector
                bbox_world = [vec3_to_list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
                obj_info = {
                    "name": obj.name,
                    "matrix_world": mat4_to_list(obj.matrix_world),
                    "bbox_world": bbox_world,
                    "data_name": obj.data.name,
                    # "uv_layout_path": os.path.abspath(uv_png_path),  # UV 示意图的绝对路径
                    "uv_layout_path": os.path.relpath(uv_png_path, out_dir) if uv_png_path.startswith(out_dir) else uv_png_path,
                    "uv_data": uv_data,
                }
                per_object_entries.append(obj_info)

            except Exception as e:
                self.report({'WARNING'}, f"{obj.name} 导出 UV 示意图失败：{e}")
                continue

        if not per_object_entries:
            self.report({'ERROR'}, "未能为任何对象导出 UV 示意图")
            return {'CANCELLED'}
        

        # -------- 写单一 metadata：Blender文件名_metadata.json --------
        meta = {
            "export_time": timestamp(),
            "render_source": mode_render,     # "CAMERA"/"VIEWPORT"
            "camera_name": camera_name,

            # 导出的一次性资源
            "image_path": image_path,         # 相机名_camera.png（绝对路径）
            "paint3d_path": paint3d_path,     # 相机名_paint_3d.png（绝对路径）

            # 目标对象列表
            "objects": per_object_entries,

            # 视口“相机”参数（尽量写，缺失则 None）
            "viewport_params": {
                "width":  int(props.viewport_width)  if mode_render == "VIEWPORT" else None,
                "height": int(props.viewport_height) if mode_render == "VIEWPORT" else None,
            },
            "viewport_intrinsics_K":          (viewport_calib["intrinsics_K"]          if viewport_calib else None),
            "viewport_projection_matrix_4x4": (viewport_calib["projection_matrix_4x4"] if viewport_calib else None),
            "viewport_world_to_view":         (viewport_calib["world_to_view"]         if viewport_calib else None),
            "viewport_view_to_world":         (viewport_calib["view_to_world"]         if viewport_calib else None),
            "viewport_R_world_to_view":       (viewport_calib["R_world_to_view"]       if viewport_calib else None),
            "viewport_t_world_to_view":       (viewport_calib["t_world_to_view"]       if viewport_calib else None),
            "viewport_view_perspective":      (viewport_calib["view_perspective"]      if viewport_calib else None),
            "viewport_lens_mm":               (viewport_calib["lens_mm"]               if viewport_calib else None),

            # 相机（若选择了相机）
            "camera_object":        (cam.name if cam else None),
            "camera_params":        cam_params,
            "camera_intrinsics_K":  K,
            "camera_to_world":      (mat4_to_list(cam_to_world) if cam_to_world else None),
            "world_to_camera":      (mat4_to_list(world_to_cam) if world_to_cam else None),
            "R_world_to_cam":       (mat3_to_list(R) if R else None),
            "t_world_to_cam":       (vec3_to_list(t) if t else None),
            "projection_matrix_4x4":proj,
        }

        meta_path = os.path.join(out_dir, f"{blend_base}_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.report({'ERROR'}, f"写入 metadata 失败：{e}")
            return {'CANCELLED'}

        # -------- 把 3 个关键路径写回到面板（绝对路径） --------
        try:
            props.meta_path = os.path.abspath(meta_path)
        except: pass
        try:
            props.paint_3d_path = os.path.abspath(paint3d_path)
        except: pass
        # UV 画布有多张，这里写回第一张，便于 UI 快速预览/修改
        try: props.paint_uv_path = ""
        except: pass

        self.report({'INFO'}, f"导出完成：图像与 3D 画布基于“{camera_name}”，{len(per_object_entries)} 个 UV 画布，元数据：{blend_base}_metadata.json")
        return {'FINISHED'}



# —— 可选：本文件内的 GPU 开启工具（独立于 apply_ops）——
def _enable_cycles_gpu_local(scene):
    try:
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
    except Exception:
        scene.cycles.device = 'CPU'
        return
    for b in ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]:
        try:
            cprefs.compute_device_type = b
            for d in cprefs.get_devices_for_type(b):
                d.use = True
            scene.cycles.device = 'GPU'
            return
        except Exception:
            continue
    scene.cycles.device = 'CPU'


class PSB_OT_RenderCameraMask(bpy.types.Operator):
    """导出选中物体/集合在【相机】视角下的遮罩（对象=白，背景=黑，纯黑白PNG）"""
    bl_idname = "psb.render_camera_mask"
    bl_label  = "导出相机视角遮罩"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.psb_exporter
        cam = props.camera
        if not cam or getattr(cam, "type", "") != 'CAMERA':
            self.report({'ERROR'}, "请先在面板选择一个相机")
            return {'CANCELLED'}

        from .utils import pick_bake_targets, ensure_dir
        targets = pick_bake_targets(props)
        if not targets:
            self.report({'ERROR'}, "未找到要导出的网格物体（请检查目标类型与目标选择）")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(props.out_dir)
        ensure_dir(out_dir)

        camera_name = cam.name
        out_path_noext = os.path.join(out_dir, f"{camera_name}_camera_mask")

        scene = context.scene
        view_layer = scene.view_layers[0]

        # —— 记录现场 —— #
        orig_cam = scene.camera
        orig_engine = scene.render.engine
        orig_filepath = scene.render.filepath
        img = scene.render.image_settings
        orig_format = img.file_format
        orig_color_mode = img.color_mode
        orig_color_depth = img.color_depth
        orig_film = scene.render.film_transparent
        orig_override = view_layer.material_override
        orig_hide_map = {obj: obj.hide_render for obj in bpy.data.objects}

        cm = scene.view_settings
        orig_view_transform = cm.view_transform
        orig_look = cm.look
        orig_exposure = cm.exposure
        orig_gamma = cm.gamma

        world = scene.world
        orig_world_use_nodes = getattr(world, "use_nodes", False) if world else False
        orig_world_color = tuple(world.color) if (world and not orig_world_use_nodes) else None
        bg_node = None
        orig_bg_col = None
        orig_bg_strength = None
        if world and orig_world_use_nodes and world.node_tree:
            bg_node = world.node_tree.nodes.get("Background")
            if bg_node:
                if "Color" in bg_node.inputs:
                    orig_bg_col = tuple(bg_node.inputs["Color"].default_value)
                if "Strength" in bg_node.inputs:
                    orig_bg_strength = float(bg_node.inputs["Strength"].default_value)

        # —— 临时“纯白”材质（不会受灯光影响） —— #
        mat = bpy.data.materials.new("PSB_Mask_PureWhite")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (200, 0)
        emi = nt.nodes.new("ShaderNodeEmission"); emi.location = (0, 0)
        emi.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # 纯白
        emi.inputs["Strength"].default_value = 1.0
        nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

        # （可选）本地启用 Cycles GPU（遮罩渲染可独立配置）
        def _enable_cycles_gpu_local(scene):
            try:
                prefs = bpy.context.preferences
                cprefs = prefs.addons['cycles'].preferences
            except Exception:
                scene.cycles.device = 'CPU'
                return
            for b in ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]:
                try:
                    cprefs.compute_device_type = b
                    for d in cprefs.get_devices_for_type(b):
                        d.use = True
                    scene.cycles.device = 'GPU'
                    return
                except Exception:
                    continue
            scene.cycles.device = 'CPU'

        try:
            scene.camera = cam
            scene.render.engine = 'CYCLES'
            if getattr(props, "use_gpu_export", getattr(props, "use_gpu", False)):
                _enable_cycles_gpu_local(scene)

            # 只让目标可渲染
            for obj in bpy.data.objects:
                obj.hide_render = (obj not in targets)

            # —— 关键：强制 Standard 视图变换，避免 Filmic 变灰 —— #
            cm.view_transform = 'Standard'
            cm.look = 'None'
            cm.exposure = 0.0
            cm.gamma = 1.0

            # 世界背景设为纯黑 + 不透明背景
            scene.render.film_transparent = False
            if world:
                if world.use_nodes and world.node_tree and bg_node:
                    if "Color" in bg_node.inputs:
                        bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                    if "Strength" in bg_node.inputs:
                        bg_node.inputs["Strength"].default_value = 1.0
                elif not world.use_nodes:
                    world.color = (0.0, 0.0, 0.0)

            # 材质覆盖 + 灰度 8bit 输出
            view_layer.material_override = mat
            scene.render.filepath = out_path_noext
            img.file_format = 'PNG'
            img.color_mode = 'BW'     # 直接灰度输出
            img.color_depth = '8'

            bpy.ops.render.render(write_still=True)
            out_path = scene.render.filepath
            if not out_path.lower().endswith(".png"):
                out_path += ".png"

        finally:
            # —— 还原现场 —— #
            for obj, hv in orig_hide_map.items():
                try: obj.hide_render = hv
                except: pass
            view_layer.material_override = orig_override

            if world:
                if world and orig_world_use_nodes and world.node_tree and bg_node:
                    if orig_bg_col is not None and "Color" in bg_node.inputs:
                        bg_node.inputs["Color"].default_value = orig_bg_col
                    if orig_bg_strength is not None and "Strength" in bg_node.inputs:
                        bg_node.inputs["Strength"].default_value = orig_bg_strength
                elif world and not orig_world_use_nodes and orig_world_color is not None:
                    world.color = orig_world_color

            scene.render.film_transparent = orig_film
            scene.render.filepath = orig_filepath
            img.file_format = orig_format
            img.color_mode = orig_color_mode
            img.color_depth = orig_color_depth

            cm.view_transform = orig_view_transform
            cm.look = orig_look
            cm.exposure = orig_exposure
            cm.gamma = orig_gamma

            scene.render.engine = orig_engine
            scene.camera = orig_cam
            try: bpy.data.materials.remove(mat, do_unlink=True)
            except: pass

        self.report({'INFO'}, f"已导出遮罩：{out_path}")
        return {'FINISHED'}



class PSB_OT_RenderCameraStill(bpy.types.Operator):
    """从选定【相机】渲染一张 PNG 到输出目录，命名为 <相机名>_camera_序号.png"""
    bl_idname = "psb.render_camera_still"
    bl_label  = "从相机渲染新图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.psb_exporter
        cam = props.camera
        if not cam or getattr(cam, "type", "") != 'CAMERA':
            self.report({'ERROR'}, "请先在面板选择一个相机")
            return {'CANCELLED'}

        from .utils import ensure_dir
        out_dir = bpy.path.abspath(props.out_dir)
        ensure_dir(out_dir)

        camera_name = cam.name

        # 计算下一个序号（从 1 开始，三位补零）
        def next_index(out_dir, cam_name):
            prefix = f"{cam_name}_camera_"
            max_idx = 0
            try:
                for fn in os.listdir(out_dir):
                    if fn.startswith(prefix) and fn.lower().endswith(".png"):
                        stem = fn[:-4]  # 去掉 .png
                        part = stem[len(prefix):]
                        if part.isdigit():
                            max_idx = max(max_idx, int(part))
            except Exception:
                pass
            return max_idx + 1

        idx = next_index(out_dir, camera_name)
        out_path_noext = os.path.join(out_dir, f"{camera_name}_camera_{idx:03d}")

        scene = context.scene
        # 记录现场
        orig_cam = scene.camera
        orig_filepath = scene.render.filepath
        img = scene.render.image_settings
        orig_format = img.file_format

        try:
            scene.camera = cam
            scene.render.filepath = out_path_noext
            img.file_format = 'PNG'
            bpy.ops.render.render(write_still=True)
            out_path = scene.render.filepath
            if not out_path.lower().endswith(".png"):
                out_path += ".png"
        finally:
            scene.camera = orig_cam
            scene.render.filepath = orig_filepath
            img.file_format = orig_format

        try: props.last_render_path = out_path
        except: pass

        self.report({'INFO'}, f"已渲染：{out_path}")
        return {'FINISHED'}
