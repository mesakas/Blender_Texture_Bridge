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
        cam    = props.camera
        obj    = props.target_object
        out_dir = bpy.path.abspath(props.out_dir)
        uv_size = props.uv_size
        mode    = props.render_mode  # "CAMERA" or "VIEWPORT"

        # ---------- 校验 ----------
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择一个网格物体")
            return {'CANCELLED'}
        if mode == "CAMERA" and (not cam or cam.type != 'CAMERA'):
            self.report({'ERROR'}, "渲染源为相机，但未选择有效相机")
            return {'CANCELLED'}

        ensure_dir(out_dir)
        base = f"{obj.name}_{timestamp()}"

        # ---------- 1) 渲染 / 抓图 ----------
        try:
            if mode == "CAMERA":
                image_path = render_from_camera(context.scene, cam, out_dir, base)
            else:
                image_path = render_from_viewport(
                    context, out_dir, base,
                    props.viewport_width, props.viewport_height
                )
        except Exception as e:
            self.report({'ERROR'}, f"渲染失败：{e}")
            return {'CANCELLED'}

        # ---------- 2) 相机参数（可选：即使是 VIEWPORT 模式，只要选了相机就导出） ----------
        K = cam_params = cam_to_world = world_to_cam = R = t = proj = None
        if cam and cam.type == 'CAMERA':
            try:
                K, cam_params = compute_camera_K(context.scene, cam)             # PERSP/ORTHO 返回 3x3；PANO 返回 None
                cam_to_world, world_to_cam, R, t = get_extrinsics(cam)           # X_cam = R * X_world + t
                proj = compute_projection_matrix_gl(context.scene, cam)          # OpenGL 风格 4x4；PANO 为 None
            except Exception as e:
                self.report({'WARNING'}, f"相机参数计算失败：{e}")

        # ---------- 2b) 视口“相机”标定（总是尝试写，便于外部对齐） ----------
        viewport_calib = None
        try:
            if mode == "VIEWPORT":
                vp_w = int(props.viewport_width)
                vp_h = int(props.viewport_height)
            else:
                # 相机模式下，沿用场景当前渲染尺寸
                vp_w = int(context.scene.render.resolution_x * (context.scene.render.resolution_percentage / 100.0))
                vp_h = int(context.scene.render.resolution_y * (context.scene.render.resolution_percentage / 100.0))
            viewport_calib = compute_viewport_calibration(context, vp_w, vp_h)
        except Exception as e:
            # 没有 3D 视口或其它异常
            viewport_calib = None

        # ---------- 3) UV 数据 + UV 轮廓 PNG ----------
        try:
            uv_data = collect_object_uv_data(obj)
            uv_png  = export_uv_layout_png(obj, out_dir, base, size=uv_size)
        except Exception as e:
            self.report({'ERROR'}, f"导出 UV 失败：{e}")
            return {'CANCELLED'}

        # ---------- 4) 物体信息 ----------
        bbox_world = [vec3_to_list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        obj_info = {
            "name": obj.name,
            "matrix_world": mat4_to_list(obj.matrix_world),
            "bbox_world": bbox_world,
            "data_name": obj.data.name,
        }

        # ---------- 5) 写 metadata ----------
        meta = {
            "export_time": timestamp(),
            "render_source": mode,  # "CAMERA" / "VIEWPORT"

            "image_path":    os.path.relpath(image_path, out_dir) if image_path.startswith(out_dir) else image_path,
            "uv_layout_path":os.path.relpath(uv_png,   out_dir) if uv_png.startswith(out_dir)   else uv_png,

            "object":  obj_info,
            "uv_data": uv_data,

            # 视口“相机”参数（总是尝试写）
            "viewport_params": {
                "width":  props.viewport_width  if mode == "VIEWPORT" else None,
                "height": props.viewport_height if mode == "VIEWPORT" else None,
            },
            "viewport_intrinsics_K":            (viewport_calib["intrinsics_K"]            if viewport_calib else None),
            "viewport_projection_matrix_4x4":   (viewport_calib["projection_matrix_4x4"]   if viewport_calib else None),
            "viewport_world_to_view":           (viewport_calib["world_to_view"]           if viewport_calib else None),
            "viewport_view_to_world":           (viewport_calib["view_to_world"]           if viewport_calib else None),
            "viewport_R_world_to_view":         (viewport_calib["R_world_to_view"]         if viewport_calib else None),
            "viewport_t_world_to_view":         (viewport_calib["t_world_to_view"]         if viewport_calib else None),
            "viewport_view_perspective":        (viewport_calib["view_perspective"]        if viewport_calib else None),
            "viewport_lens_mm":                 (viewport_calib["lens_mm"]                 if viewport_calib else None),

            # 相机（若选择了相机）
            "camera_object":        (cam.name if cam else None),
            "camera_params":        cam_params,                             # 分辨率、传感器、镜头等
            "camera_intrinsics_K":  K,                                       # PERSP/ORTHO 为 3x3；PANO 为 None
            "camera_to_world":      (mat4_to_list(cam_to_world) if cam_to_world else None),
            "world_to_camera":      (mat4_to_list(world_to_cam) if world_to_cam else None),
            "R_world_to_cam":       (mat3_to_list(R) if R else None),
            "t_world_to_cam":       (vec3_to_list(t) if t else None),
            "projection_matrix_4x4":proj,                                    # OpenGL 风格 4x4；PANO 为 None
        }

        meta_path = os.path.join(out_dir, f"{base}_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

                # 在成功写入 meta_path 后追加：
                props.meta_path = bpy.path.relpath(meta_path)
                
        except Exception as e:
            self.report({'ERROR'}, f"写入 metadata 失败：{e}")
            return {'CANCELLED'}



        # ---------- 6) 生成两张空白图片 ----------
        # 尺寸：UV
        uv_canvas_path = os.path.join(out_dir, f"{obj.name}_paint_uv.png")
        create_blank_png(uv_size, uv_size, uv_canvas_path)
        props.paint_uv_path = bpy.path.relpath(uv_canvas_path)

        # 尺寸：渲染输出
        if mode == "VIEWPORT":
            rw, rh = int(props.viewport_width), int(props.viewport_height)
        else:
            s = context.scene
            rw = int(s.render.resolution_x * (s.render.resolution_percentage / 100.0))
            rh = int(s.render.resolution_y * (s.render.resolution_percentage / 100.0))
        paint3d_path = os.path.join(out_dir, f"{obj.name}_paint_3d.png")
        create_blank_png(rw, rh, paint3d_path)
        props.paint_3d_path = bpy.path.relpath(paint3d_path)


        self.report({'INFO'}, f"导出完成：{out_dir}")
        return {'FINISHED'}
