# apply_ops.py
import os
import bpy
from mathutils import Matrix
from .utils import (
    ensure_dir,
    create_blank_png,
    load_image_from_path,
    assign_image_to_basecolor,
    ensure_uv_map,
    get_active_uv_image_size,
    compute_viewport_calibration,
    pick_bake_targets,
    to_abs, make_outpath, copy_backup, alpha_over, 
    timestamp,
)


import json



def _enable_cycles_gpu(scene):
    """
    把 Cycles 尽可能切到 GPU（优先 OPTIX/CUDA/HIP/METAL/ONEAPI），
    无可用设备则回落 CPU。静默容错。
    """
    import bpy
    prefs = bpy.context.preferences
    try:
        cprefs = prefs.addons['cycles'].preferences
    except Exception:
        scene.cycles.device = 'CPU'
        return

    backends = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
    ok = False
    for b in backends:
        try:
            cprefs.compute_device_type = b
            # 启用该类型下的所有设备
            for d in cprefs.get_devices_for_type(b):
                d.use = True
            scene.cycles.device = 'GPU'
            ok = True
            break
        except Exception:
            continue

    if not ok:
        scene.cycles.device = 'CPU'

def _render_depth_map(context, cam_obj, width, height, out_path, enable_gpu=False):
    """
    使用 Cycles + ViewLayer.material_override 渲染一张“到相机的线性距离图”。
    输出 OpenEXR 32-bit BW。
    """
    import bpy
    scene = context.scene
    view_layer = scene.view_layers[0]

    # --- 记录现场 ---
    orig_cam = scene.camera
    orig_rx  = scene.render.resolution_x
    orig_ry  = scene.render.resolution_y
    orig_rp  = scene.render.resolution_percentage
    orig_engine = scene.render.engine
    orig_filepath = scene.render.filepath
    img_set = scene.render.image_settings
    orig_format = img_set.file_format
    orig_color_mode = img_set.color_mode
    orig_color_depth = img_set.color_depth
    orig_exr_codec = getattr(img_set, "exr_codec", "ZIP")
    orig_override = view_layer.material_override
    orig_film_transparent = scene.render.film_transparent
    cm = scene.view_settings
    orig_view_transform = cm.view_transform
    orig_look = cm.look
    orig_exposure = cm.exposure
    orig_gamma = cm.gamma

    # --- 覆盖材质（输出distance） ---
    depth_mat = bpy.data.materials.new("PSB_DepthOverrideMat")
    depth_mat.use_nodes = True
    nt = depth_mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (400, 0)
    geo = nt.nodes.new("ShaderNodeNewGeometry");    geo.location = (0, -160)

    cam_loc = cam_obj.matrix_world.translation
    camX = nt.nodes.new("ShaderNodeValue"); camX.location = (0, -380); camX.outputs[0].default_value = float(cam_loc.x)
    camY = nt.nodes.new("ShaderNodeValue"); camY.location = (0, -420); camY.outputs[0].default_value = float(cam_loc.y)
    camZ = nt.nodes.new("ShaderNodeValue"); camZ.location = (0, -460); camZ.outputs[0].default_value = float(cam_loc.z)
    cmb  = nt.nodes.new("ShaderNodeCombineXYZ");    cmb.location  = (180, -420)
    nt.links.new(camX.outputs[0], cmb.inputs[0])
    nt.links.new(camY.outputs[0], cmb.inputs[1])
    nt.links.new(camZ.outputs[0], cmb.inputs[2])

    vsub = nt.nodes.new("ShaderNodeVectorMath");    vsub.location = (360, -260); vsub.operation = 'SUBTRACT'
    nt.links.new(cmb.outputs["Vector"],   vsub.inputs[0])
    nt.links.new(geo.outputs["Position"], vsub.inputs[1])
    vlen = nt.nodes.new("ShaderNodeVectorMath");    vlen.location = (540, -260); vlen.operation = 'LENGTH'
    nt.links.new(vsub.outputs["Vector"], vlen.inputs[0])
    nt.links.new(vlen.outputs["Value"], emi.inputs["Color"])
    emi.inputs["Strength"].default_value = 1.0
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

    # --- 应用并渲染 ---
    try:
        scene.camera = cam_obj
        scene.render.engine = 'CYCLES'
        if enable_gpu:
            _enable_cycles_gpu(scene)

        scene.render.resolution_x = int(width)
        scene.render.resolution_y = int(height)
        scene.render.resolution_percentage = 100
        scene.render.filepath = os.path.splitext(out_path)[0]
        img_set.file_format = 'OPEN_EXR'
        img_set.color_mode  = 'BW'
        img_set.color_depth = '32'
        if hasattr(img_set, "exr_codec"):
            img_set.exr_codec = 'ZIP'
        scene.render.film_transparent = True

        cm.view_transform = 'Standard'
        cm.look = 'None'
        cm.exposure = 0.0
        cm.gamma = 1.0

        view_layer.material_override = depth_mat
        bpy.ops.render.render(write_still=True)

        saved = scene.render.filepath
        if not saved.lower().endswith(".exr"):
            saved = saved + ".exr"

    finally:
        view_layer.material_override = orig_override
        scene.render.filepath = orig_filepath
        scene.render.engine = orig_engine
        scene.render.resolution_x = orig_rx
        scene.render.resolution_y = orig_ry
        scene.render.resolution_percentage = orig_rp
        img_set.file_format = orig_format
        img_set.color_mode  = orig_color_mode
        img_set.color_depth = orig_color_depth
        if hasattr(img_set, "exr_codec"):
            img_set.exr_codec = orig_exr_codec
        scene.camera = orig_cam
        scene.render.film_transparent = orig_film_transparent

        cm.view_transform = orig_view_transform
        cm.look = orig_look
        cm.exposure = orig_exposure
        cm.gamma = orig_gamma

        try: bpy.data.materials.remove(depth_mat, do_unlink=True)
        except: pass

    return saved

def _make_frontfacing_vgroup(obj, cam_obj, name="PSB_FrontFacing", threshold_deg=0.0):
    """
    创建/覆盖一个临时顶点组：只包含“朝向相机”的面（按阈值）的顶点。
    - threshold_deg: 允许的最小正面角度（度）。0 表示严格正面（法线与视线夹角 < 90° 即通过）。
    返回创建的顶点组名称。
    """
    import math
    me = obj.data
    # 确保有这个顶点组（没有就新建）
    vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)

    # 先把所有顶点权重清零
    all_idx = [v.index for v in me.vertices]
    if all_idx:
        vg.add(all_idx, 0.0, 'REPLACE')

    mw = obj.matrix_world
    cam_loc = cam_obj.matrix_world.translation

    cos_thresh = math.cos(math.radians(90.0 - threshold_deg))  # 等价于 dot > 0 当 threshold_deg=0

    # 给“朝向相机”的面的顶点赋值 1.0
    for poly in me.polygons:
        # 面中心与法线（世界空间）
        n_world = (mw.to_3x3() @ poly.normal).normalized()
        center_world = mw @ poly.center
        view_dir = (cam_loc - center_world).normalized()

        # dot > 0 表示朝向相机；可加阈值（略微收紧）
        if n_world.dot(view_dir) > 0.0:  # 如需更严格：> cos_thresh
            for vi in poly.vertices:
                vg.add([vi], 1.0, 'REPLACE')

    return vg.name

# —— 从 JSON 建立相机：完全按导出参数还原 —— #
def _camera_from_json(context, meta: dict, paint_w: int, paint_h: int):
    """
    返回 (cam_obj, cam_data, used_mode)
    - 优先使用 meta['camera_*']（相机导出）
    - 否则回退到 meta['viewport_*']（视口导出）
    - 都没有则抛异常
    """
    scene = context.scene

    # 1) 优先相机
    K = meta.get("camera_intrinsics_K")
    cam_params = meta.get("camera_params")
    cam_to_world = meta.get("camera_to_world")
    if K and cam_params and cam_to_world:
        cam = bpy.data.cameras.new("PSB_JSONCamera")
        cam_obj = bpy.data.objects.new("PSB_JSONCamera", cam)
        context.collection.objects.link(cam_obj)

        # 设置镜头/传感器
        cam.type = 'PERSP' if cam_params.get("type","PERSP") != "ORTHO" else 'ORTHO'
        cam.sensor_fit = cam_params.get("sensor_fit", "AUTO")
        cam.sensor_width  = float(cam_params.get("sensor_width_mm", 36.0))
        cam.sensor_height = float(cam_params.get("sensor_height_mm", 24.0))
        cam.lens = float(cam_params.get("lens_mm", 50.0))

        # 主点偏移（近似）：Blender shift 以传感器宽/高为单位的相对偏移
        # 归一化近似：cx,cy 以像素计，中心=width/2,height/2
        # shift_x ≈ (0.5 - cx/width), shift_y ≈ (cy/height - 0.5)
        cx = float(K[0][2]); cy = float(K[1][2])
        cx_n = cx / float(cam_params["resolution_x"])
        cy_n = cy / float(cam_params["resolution_y"])
        cam.shift_x = (0.5 - cx_n)
        cam.shift_y = (cy_n - 0.5)

        # 位姿
        M = Matrix([[cam_to_world[i][j] for j in range(4)] for i in range(4)])
        cam_obj.matrix_world = M

        # 裁剪面（若未提供，保守值）
        cam.clip_start = 0.01
        cam.clip_end   = 10000.0

        return cam_obj, cam, "CAMERA_JSON"

    # 2) 视口回退
    PV = meta.get("viewport_projection_matrix_4x4")
    V  = meta.get("viewport_world_to_view")
    v2w = meta.get("viewport_view_to_world")
    Kvp = meta.get("viewport_intrinsics_K")
    if Kvp and v2w:
        cam = bpy.data.cameras.new("PSB_JSONViewportCam")
        cam_obj = bpy.data.objects.new("PSB_JSONViewportCam", cam)
        context.collection.objects.link(cam_obj)

        # 视口透视/正交
        view_persp = meta.get("viewport_view_perspective", "PERSP")
        cam.type = 'PERSP' if view_persp != "ORTHO" else 'ORTHO'

        # 我们需要一个自洽的（sensor, lens, shift）使得像素内参匹配：
        # 选择 sensor_width=36mm, sensor_height=36mm（正方形“胶片”），
        # lens_mm 由 fx 反推： fx = f * width / sensor_width  => f = fx * sensor_width / width
        # 再用 cx, cy 设置 shift
        cam.sensor_width = 36.0
        cam.sensor_height = 36.0
        fx = float(Kvp[0][0]); fy = float(Kvp[1][1])
        width  = float(meta.get("viewport_params",{}).get("width")  or paint_w)
        height = float(meta.get("viewport_params",{}).get("height") or paint_h)
        cam.lens = max(1e-3, fx * cam.sensor_width / max(1.0, width))

        cx = float(Kvp[0][2]); cy = float(Kvp[1][2])
        cam.shift_x = (0.5 - cx/width)
        cam.shift_y = (cy/height - 0.5)

        # 位姿
        M = Matrix([[v2w[i][j] for j in range(4)] for i in range(4)])
        cam_obj.matrix_world = M
        cam.clip_start = 0.01
        cam.clip_end   = 10000.0

        return cam_obj, cam, "VIEWPORT_JSON"

    raise RuntimeError("元数据中缺少可用于重建相机/视口的参数（需要 camera_* 或 viewport_*）。")

# —— 用 JSON 相机做烘焙 —— #
class PSB_OT_ApplyPaint3D(bpy.types.Operator):
    bl_idname = "psb.apply_paint3d"
    bl_label  = "应用 3D 绘制 → 贴图（按JSON）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import math, json, os
        p = context.scene.psb_exporter
        targets = pick_bake_targets(p)
        if not targets:
            self.report({'ERROR'}, "未找到要处理的网格物体（请检查目标类型与目标选择）")
            return {'CANCELLED'}

        if not p.paint_3d_path:
            self.report({'ERROR'}, "未设置 3D 绘制图片路径"); return {'CANCELLED'}

        # === 输出目录（带会话时间戳子目录） ===
        out_dir  = to_abs(p.out_dir) if p.out_dir else os.getcwd()
        bake_dir = to_abs(getattr(p, "bake_dir", "")) if getattr(p, "bake_dir", "") else os.path.join(out_dir, "baked")
        ensure_dir(bake_dir)
        session_dir = os.path.join(bake_dir, f"backed_{timestamp()}")
        ensure_dir(session_dir)

        use_gpu_depth = bool(getattr(p, "use_gpu_depth", getattr(p, "use_gpu", False)))
        use_gpu_bake  = bool(getattr(p, "use_gpu_bake",  getattr(p, "use_gpu", False)))

        # === 读取 JSON ===
        meta_path = to_abs(p.meta_path)
        if not meta_path or not os.path.exists(meta_path):
            self.report({'ERROR'}, "未找到 metadata JSON，请在面板指定"); return {'CANCELLED'}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"读取 JSON 失败：{e}"); return {'CANCELLED'}

        # === 源图 ===
        src_path = to_abs(p.paint_3d_path)
        if not src_path or not os.path.exists(src_path):
            self.report({'ERROR'}, f"找不到 3D 绘制图片：{src_path}"); return {'CANCELLED'}
        img_src = load_image_from_path(src_path)
        if not img_src:
            self.report({'ERROR'}, f"无法加载 3D 绘制图片：{src_path}"); return {'CANCELLED'}
        try: img_src.reload()
        except: pass
        if hasattr(img_src, "colorspace_settings"):
            img_src.colorspace_settings.name = 'sRGB'
        try:
            img_src.use_mipmap = False
            img_src.use_interpolation = True
        except: pass
        paint_w, paint_h = int(img_src.size[0]), int(img_src.size[1])

        # === 相机重建 ===
        try:
            cam_obj, cam_data, used = _camera_from_json(context, meta, paint_w, paint_h)
        except Exception as e:
            self.report({'ERROR'}, f"重建相机失败：{e}"); return {'CANCELLED'}

        # === 深度 EXR（一次） ===
        if use_gpu_depth:
            _enable_cycles_gpu(context.scene)

        depth_exr_path = make_outpath(session_dir, "depth", targets[0].name, "exr")
        saved_depth = _render_depth_map(
            context, cam_obj, paint_w, paint_h, depth_exr_path, enable_gpu=use_gpu_depth
        )
        depth_img = load_image_from_path(saved_depth)
        if not depth_img:
            self.report({'ERROR'}, "深度图渲染失败"); return {'CANCELLED'}
        if hasattr(depth_img, "colorspace_settings"):
            depth_img.colorspace_settings.name = 'Non-Color'
        try:
            depth_img.use_mipmap = False
            depth_img.use_interpolation = False
        except: pass

        # 背景色（RGBA）
        bg_rgba = tuple(getattr(p, "bake_background_color", (1.0, 1.0, 1.0, 0.0)))
        bg_r, bg_g, bg_b, bg_a = float(bg_rgba[0]), float(bg_rgba[1]), float(bg_rgba[2]), float(bg_rgba[3])

        # 深度容差参数
        eps_rel = float(getattr(p, "depth_epsilon_ratio", 5e-5) or 5e-5)      # 相对深度
        eps_min = float(getattr(p, "depth_epsilon_min", 1e-5) or 1e-5)        # 下限
        eps_max_ratio = float(getattr(p, "depth_epsilon_max_ratio", 0.002) or 0.002)  # 上限相对对角线

        baked_count = 0
        last_baked_path = ""
        log_lines = []
        log_lines.append(f"ApplyPaint3D session @ {timestamp()}")
        log_lines.append(f"meta: {meta_path}")
        log_lines.append(f"paint3d: {src_path}")
        log_lines.append(f"depth_exr: {saved_depth}")

        try:
            for obj in targets:
                # 目标贴图尺寸
                target_w, target_h = get_active_uv_image_size(obj)
                if not target_w:
                    target_w = target_h = int(p.uv_size)

                # 临时接收图像（颜色与覆盖率）
                paint_img = bpy.data.images.new(name=f"{obj.name}_PaintOnly", width=target_w, height=target_h, alpha=False)
                if hasattr(paint_img, "colorspace_settings"):
                    paint_img.colorspace_settings.name = 'sRGB'
                mask_img  = bpy.data.images.new(name=f"{obj.name}_MaskOnly",  width=target_w, height=target_h, alpha=False)
                if hasattr(mask_img, "colorspace_settings"):
                    mask_img.colorspace_settings.name = 'Non-Color'

                # 最终图像
                target_img = bpy.data.images.new(name=f"{obj.name}_Baked", width=target_w, height=target_h, alpha=True)
                try:
                    target_img.alpha_mode = 'STRAIGHT'
                    target_img.use_alpha = True
                    if hasattr(target_img, "colorspace_settings"):
                        target_img.colorspace_settings.name = 'sRGB'
                    target_img.pixels[:] = [bg_r, bg_g, bg_b, bg_a] * (target_w * target_h)
                except: pass

                # UV Project
                proj_uv = ensure_uv_map(obj, "PSB_ProjectedUV")
                mod = obj.modifiers.new("PSB_UVProject", type='UV_PROJECT')
                mod.uv_layer = proj_uv
                proj = mod.projectors
                if len(proj) < 1: proj.add()
                proj[0].object = cam_obj
                mod.aspect_x = float(paint_w) / max(1.0, float(paint_h))
                mod.aspect_y = 1.0

                # 备份材质
                orig_mats = [m for m in obj.data.materials]

                # —— 节点：输出“Tex × (mask × alpha)”颜色，并单独可烘焙覆盖率 —— #
                mat = bpy.data.materials.new(name="PSB_TempBakeMat")
                mat.use_nodes = True
                nt = mat.node_tree
                for n in list(nt.nodes): nt.nodes.remove(n)

                out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1150, 0)
                emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (950, 0)

                tex = nt.nodes.new("ShaderNodeTexImage");       tex.location = (40, 0)
                tex.image = img_src
                tex.projection = 'FLAT'
                tex.extension  = 'EXTEND'   # 关键：避免越界采黑
                tex.interpolation = 'Linear'

                uvn = nt.nodes.new("ShaderNodeUVMap");          uvn.location = (-220, -150); uvn.uv_map = proj_uv
                nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

                depth_tex = nt.nodes.new("ShaderNodeTexImage"); depth_tex.location = (40, -300)
                depth_tex.image = depth_img
                depth_tex.interpolation = 'Closest'
                depth_tex.projection = 'FLAT'
                depth_tex.extension  = 'CLIP'
                if hasattr(depth_tex, "color_space"): depth_tex.color_space = 'NONE'
                nt.links.new(uvn.outputs["UV"], depth_tex.inputs["Vector"])

                # 可见性（正面 × 深度一致）
                geo = nt.nodes.new("ShaderNodeNewGeometry");        geo.location = (-520, -420)
                pos_xf = nt.nodes.new("ShaderNodeVectorTransform"); pos_xf.location = (-320, -420)
                pos_xf.vector_type = 'POINT'; pos_xf.convert_from = 'OBJECT'; pos_xf.convert_to = 'WORLD'
                nt.links.new(geo.outputs["Position"], pos_xf.inputs["Vector"])
                nrm_xf = nt.nodes.new("ShaderNodeVectorTransform"); nrm_xf.location = (-320, -520)
                nrm_xf.vector_type = 'NORMAL'; nrm_xf.convert_from = 'OBJECT'; nrm_xf.convert_to = 'WORLD'
                nt.links.new(geo.outputs["Normal"], nrm_xf.inputs["Vector"])

                cam_loc = cam_obj.matrix_world.translation
                camX = nt.nodes.new("ShaderNodeValue"); camX.outputs[0].default_value = float(cam_loc.x); camX.location = (-280, -640)
                camY = nt.nodes.new("ShaderNodeValue"); camY.outputs[0].default_value = float(cam_loc.y); camY.location = (-280, -680)
                camZ = nt.nodes.new("ShaderNodeValue"); camZ.outputs[0].default_value = float(cam_loc.z); camZ.location = (-280, -720)
                cmb  = nt.nodes.new("ShaderNodeCombineXYZ");       cmb.location  = (-100,  -680)
                nt.links.new(camX.outputs[0], cmb.inputs[0]); nt.links.new(camY.outputs[0], cmb.inputs[1]); nt.links.new(camZ.outputs[0], cmb.inputs[2])

                vsub = nt.nodes.new("ShaderNodeVectorMath"); vsub.location = (100, -520); vsub.operation = 'SUBTRACT'
                nt.links.new(cmb.outputs["Vector"], vsub.inputs[0]); nt.links.new(pos_xf.outputs["Vector"], vsub.inputs[1])
                vnorm = nt.nodes.new("ShaderNodeVectorMath"); vnorm.location = (280, -520); vnorm.operation = 'NORMALIZE'
                nt.links.new(vsub.outputs["Vector"], vnorm.inputs[0])
                vdot = nt.nodes.new("ShaderNodeVectorMath"); vdot.location = (460, -520); vdot.operation = 'DOT_PRODUCT'
                nt.links.new(nrm_xf.outputs["Vector"], vdot.inputs[0]); nt.links.new(vnorm.outputs["Vector"], vdot.inputs[1])

                # 正面阈值（角度） × 严格 dot>0 双门
                tdeg = float(getattr(p, "mask_front_threshold", 0.5) or 0.5)
                cos_thresh = math.cos(math.radians(90.0 - tdeg))
                gate_soft = nt.nodes.new("ShaderNodeMath"); gate_soft.location = (640, -520); gate_soft.operation = 'GREATER_THAN'
                gate_soft.inputs[1].default_value = cos_thresh
                nt.links.new(vdot.outputs["Value"], gate_soft.inputs[0])

                gate_strict = nt.nodes.new("ShaderNodeMath"); gate_strict.location = (820, -520); gate_strict.operation = 'GREATER_THAN'
                gate_strict.inputs[1].default_value = 0.0
                nt.links.new(vdot.outputs["Value"], gate_strict.inputs[0])

                face_gate = nt.nodes.new("ShaderNodeMath"); face_gate.location = (1000, -520); face_gate.operation = 'MULTIPLY'
                nt.links.new(gate_soft.outputs["Value"],   face_gate.inputs[0])
                nt.links.new(gate_strict.outputs["Value"], face_gate.inputs[1])

                # 可选：仅反转“正面判定”一项
                face_gate_socket = face_gate.outputs["Value"]
                if getattr(p, "mask_invert_facing", False):
                    inv = nt.nodes.new("ShaderNodeMath"); inv.location = (1180, -520); inv.operation = 'SUBTRACT'
                    inv.inputs[0].default_value = 1.0
                    nt.links.new(face_gate.outputs["Value"], inv.inputs[1])
                    face_gate_socket = inv.outputs["Value"]

                # 与深度一致性
                dist = nt.nodes.new("ShaderNodeVectorMath"); dist.location = (460, -320); dist.operation = 'LENGTH'
                nt.links.new(vsub.outputs["Vector"], dist.inputs[0])

                # 计算每像素动态 epsilon： clamp(max(depth*eps_rel, eps_min), 0, eps_max)
                # eps_max 依据物体对角线
                bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
                if bbox:
                    import mathutils
                    mins = mathutils.Vector((min(pv.x for pv in bbox), min(pv.y for pv in bbox), min(pv.z for pv in bbox)))
                    maxs = mathutils.Vector((max(pv.x for pv in bbox), max(pv.y for pv in bbox), max(pv.z for pv in bbox)))
                    diag = (maxs - mins).length
                else:
                    diag = 1.0
                eps_max = max(1e-6, eps_max_ratio * diag)

                val_rel = nt.nodes.new("ShaderNodeValue"); val_rel.location = (220, -300); val_rel.outputs[0].default_value = eps_rel
                mul_rel = nt.nodes.new("ShaderNodeMath");  mul_rel.location = (340, -300); mul_rel.operation = 'MULTIPLY'
                nt.links.new(depth_tex.outputs["Color"], mul_rel.inputs[0])
                nt.links.new(val_rel.outputs[0],         mul_rel.inputs[1])

                val_min = nt.nodes.new("ShaderNodeValue"); val_min.location = (520, -260); val_min.outputs[0].default_value = eps_min
                eps_max_v = nt.nodes.new("ShaderNodeValue"); eps_max_v.location = (520, -300); eps_max_v.outputs[0].default_value = eps_max

                eps_lo = nt.nodes.new("ShaderNodeMath"); eps_lo.location = (640, -300); eps_lo.operation = 'MAXIMUM'
                nt.links.new(mul_rel.outputs["Value"], eps_lo.inputs[0])
                nt.links.new(val_min.outputs[0],        eps_lo.inputs[1])

                eps_dyn = nt.nodes.new("ShaderNodeMath"); eps_dyn.location = (820, -300); eps_dyn.operation = 'MINIMUM'
                nt.links.new(eps_lo.outputs["Value"], eps_dyn.inputs[0])
                nt.links.new(eps_max_v.outputs[0],    eps_dyn.inputs[1])

                add_eps = nt.nodes.new("ShaderNodeMath"); add_eps.location = (1000, -300); add_eps.operation = 'ADD'
                nt.links.new(depth_tex.outputs["Color"], add_eps.inputs[0])
                nt.links.new(eps_dyn.outputs["Value"],  add_eps.inputs[1])

                depth_ok = nt.nodes.new("ShaderNodeMath"); depth_ok.location = (1180, -300); depth_ok.operation = 'LESS_THAN'
                nt.links.new(dist.outputs["Value"], depth_ok.inputs[0]); nt.links.new(add_eps.outputs["Value"], depth_ok.inputs[1])

                # === UV in [0,1] 门（你的做法保留）===
                sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = ( -40, -60)
                nt.links.new(uvn.outputs["UV"], sep.inputs["Vector"])
                u_ge_0 = nt.nodes.new("ShaderNodeMath"); u_ge_0.location = (140, -40);  u_ge_0.operation = 'GREATER_THAN'; u_ge_0.inputs[1].default_value = 0.0; nt.links.new(sep.outputs["X"], u_ge_0.inputs[0])
                u_le_1 = nt.nodes.new("ShaderNodeMath"); u_le_1.location = (140, -90);  u_le_1.operation = 'LESS_THAN';   u_le_1.inputs[1].default_value = 1.0; nt.links.new(sep.outputs["X"], u_le_1.inputs[0])
                v_ge_0 = nt.nodes.new("ShaderNodeMath"); v_ge_0.location = (140, -160); v_ge_0.operation = 'GREATER_THAN'; v_ge_0.inputs[1].default_value = 0.0; nt.links.new(sep.outputs["Y"], v_ge_0.inputs[0])
                v_le_1 = nt.nodes.new("ShaderNodeMath"); v_le_1.location = (140, -210); v_le_1.operation = 'LESS_THAN';   v_le_1.inputs[1].default_value = 1.0; nt.links.new(sep.outputs["Y"], v_le_1.inputs[0])
                uv_and1 = nt.nodes.new("ShaderNodeMath"); uv_and1.location = (320, -80);  uv_and1.operation = 'MULTIPLY'; nt.links.new(u_ge_0.outputs["Value"], uv_and1.inputs[0]); nt.links.new(u_le_1.outputs["Value"], uv_and1.inputs[1])
                uv_and2 = nt.nodes.new("ShaderNodeMath"); uv_and2.location = (320, -190); uv_and2.operation = 'MULTIPLY'; nt.links.new(v_ge_0.outputs["Value"], uv_and2.inputs[0]); nt.links.new(v_le_1.outputs["Value"], uv_and2.inputs[1])
                uv_in01 = nt.nodes.new("ShaderNodeMath"); uv_in01.location = (520, -140); uv_in01.operation = 'MULTIPLY'; nt.links.new(uv_and1.outputs["Value"], uv_in01.inputs[0]); nt.links.new(uv_and2.outputs["Value"], uv_in01.inputs[1])

                # 最终可见性
                vis_and = nt.nodes.new("ShaderNodeMath"); vis_and.location = (1360, -360); vis_and.operation = 'MULTIPLY'
                nt.links.new(face_gate_socket,           vis_and.inputs[0])
                vis_and2 = nt.nodes.new("ShaderNodeMath"); vis_and2.location = (1520, -360); vis_and2.operation = 'MULTIPLY'
                nt.links.new(depth_ok.outputs["Value"],  vis_and2.inputs[0])
                nt.links.new(uv_in01.outputs["Value"],   vis_and2.inputs[1])

                mask_final = nt.nodes.new("ShaderNodeMath"); mask_final.location = (1680, -360); mask_final.operation = 'MULTIPLY'
                nt.links.new(vis_and.inputs["Value"],  mask_final.inputs[0])
                nt.links.new(vis_and2.outputs["Value"], mask_final.inputs[1])

                # 覆盖率 = mask_final × TexAlpha
                fac_mul = nt.nodes.new("ShaderNodeMath"); fac_mul.location = (780, -160); fac_mul.operation = 'MULTIPLY'
                nt.links.new(mask_final.outputs["Value"], fac_mul.inputs[0])
                nt.links.new(tex.outputs["Alpha"],       fac_mul.inputs[1])

                # ===== 第一次 Bake：颜色×(覆盖率) 到 paint_img =====
                mix_tex = nt.nodes.new("ShaderNodeMixRGB"); mix_tex.location = (780, 0); mix_tex.blend_type = 'MIX'
                mix_tex.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
                nt.links.new(tex.outputs["Color"], mix_tex.inputs["Color2"])
                nt.links.new(fac_mul.outputs["Value"], mix_tex.inputs["Fac"])

                nt.links.new(mix_tex.outputs["Color"],  emi.inputs["Color"])
                emi.inputs["Strength"].default_value = 1.0
                nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

                paint_node = nt.nodes.new("ShaderNodeTexImage"); paint_node.location = (260, -240)
                paint_node.image = paint_img
                nt.nodes.active = paint_node

                # 绑定材质
                if not obj.data.materials: obj.data.materials.append(mat)
                else: obj.data.materials[0] = mat

                scene = context.scene
                orig_engine = scene.render.engine
                try:
                    scene.render.engine = 'CYCLES'
                    if use_gpu_bake:
                        _enable_cycles_gpu(scene)

                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True); context.view_layer.objects.active = obj

                    if hasattr(scene, "cycles"): scene.cycles.samples = 1
                    scene.render.bake.use_clear = True
                    scene.render.bake.use_selected_to_active = False
                    scene.render.bake.margin = max(8, int(0.004 * max(target_w, target_h)))
                    bpy.ops.object.bake(type='EMIT')

                    # ===== 第二次 Bake：覆盖率 fac_mul 到 mask_img =====
                    mask_node = nt.nodes.new("ShaderNodeTexImage"); mask_node.location = (260, -520)
                    mask_node.image = mask_img
                    nt.nodes.active = mask_node

                    # 让发光直接输出 fac_mul
                    for link in list(nt.links):
                        if link.to_node == emi and link.to_socket.name == "Color":
                            nt.links.remove(link)
                    nt.links.new(fac_mul.outputs["Value"], emi.inputs["Color"])

                    scene.render.bake.use_clear = True
                    scene.render.bake.margin = 2
                    bpy.ops.object.bake(type='EMIT')

                    # ===== Python 合成：背景与连续 Alpha =====
                    col_src = list(paint_img.pixels[:])  # RGBA（RGB=tex*fac_mul）
                    msk_src = list(mask_img.pixels[:])   # RGBA（R=fac_mul）
                    px = target_w * target_h
                    out_buf = [0.0] * (px * 4)
                    for i in range(px):
                        j = i * 4
                        m = msk_src[j]
                        if m < 0.0: m = 0.0
                        elif m > 1.0: m = 1.0
                        r_tex = col_src[j]
                        g_tex = col_src[j+1]
                        b_tex = col_src[j+2]
                        out_buf[j]   = bg_r * (1.0 - m) + r_tex
                        out_buf[j+1] = bg_g * (1.0 - m) + g_tex
                        out_buf[j+2] = bg_b * (1.0 - m) + b_tex
                        out_buf[j+3] = m + (1.0 - m) * bg_a
                    target_img.pixels[:] = out_buf

                    # ===== 保存 =====
                    baked_path = os.path.join(session_dir, f"baked_{obj.name}.png")
                    target_img.filepath_raw = baked_path
                    target_img.file_format = 'PNG'
                    target_img.save()

                    last_baked_path = baked_path
                    baked_count += 1
                    log_lines.append(f"baked: {baked_path}")

                    # 清理临时节点与图像
                    try:
                        nt.nodes.remove(paint_node)
                        nt.nodes.remove(mask_node)
                    except: pass
                    try:
                        bpy.data.images.remove(paint_img, do_unlink=True)
                        bpy.data.images.remove(mask_img,  do_unlink=True)
                    except: pass

                finally:
                    try: obj.modifiers.remove(mod)
                    except: pass
                    try:
                        obj.data.materials.clear()
                        for m in orig_mats:
                            obj.data.materials.append(m)
                    except: pass
                    try: bpy.data.materials.remove(mat, do_unlink=True)
                    except: pass
                    scene.render.engine = orig_engine

        finally:
            # 一次性资源清理
            if cam_data is not None:
                try: bpy.data.objects.remove(cam_obj, do_unlink=True)
                except: pass
                try: bpy.data.cameras.remove(cam_data, do_unlink=True)
                except: pass

        if last_baked_path:
            p.baked_path = last_baked_path  # 绝对路径

        # 写日志
        try:
            log_path = os.path.join(session_dir, f"log_{targets[0].name}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
        except Exception:
            pass

        self.report({'INFO'}, f"按 JSON 投影并烘焙完成：{baked_count} 个对象（模式：{used}）")
        return {'FINISHED'}


def _make_temp_projector_camera(context, mode: str, camera_obj, vp_calib, width:int, height:int):
    """
    返回 (cam_obj, cam_data)
    - CAMERA: 直接用现有相机
    - VIEWPORT: 若未提供 vp_calib，现场 compute；用视口标定重建临时相机
    """
    from .utils import compute_viewport_calibration
    scene = context.scene

    # 相机模式：直接返回现有相机
    if mode == "CAMERA" and camera_obj and getattr(camera_obj, "type", "") == 'CAMERA':
        return camera_obj, None  # (相机对象, 无需清理)

    # 视口模式：确保拿到 vp_calib
    if mode == "VIEWPORT":
        if vp_calib is None:
            try:
                vp_calib = compute_viewport_calibration(context, int(width), int(height))
            except Exception as e:
                raise RuntimeError(f"无法从当前视口获取标定：{e}")
        if not isinstance(vp_calib, dict):
            raise RuntimeError("视口标定数据无效（期望 dict）。")

        # 创建临时相机
        cam = bpy.data.cameras.new("PSB_TempProjectorCam")
        cam_obj = bpy.data.objects.new("PSB_TempProjectorCam", cam)
        context.collection.objects.link(cam_obj)

        view_persp = vp_calib.get("view_perspective", "PERSP")
        if view_persp == 'ORTHO':
            cam.type = 'ORTHO'
            cam.ortho_scale = 1.0  # 粗略值；若需要可基于 K 反推更准的 scale
        else:
            cam.type = 'PERSP'

        # 根据内参 K 设置 lens 与 shift
        K = vp_calib.get("intrinsics_K")
        if K:
            fx = float(K[0][0]); fy = float(K[1][1])
            cx = float(K[0][2]); cy = float(K[1][2])
            W  = float(width);   H  = float(height)

            # 选定一个“虚拟胶片宽度”，用 fx 反推 lens（针孔近似）
            cam.sensor_width = 36.0
            cam.sensor_height = 36.0
            cam.lens = max(1e-3, fx * cam.sensor_width / max(1.0, W))

            # 主点偏移 → Blender 的 shift（归一化近似）
            cam.shift_x = (0.5 - cx / W)
            cam.shift_y = (cy / H - 0.5)

        # 位姿：用 view_to_world
        v2w = vp_calib.get("view_to_world")
        if v2w:
            from mathutils import Matrix
            M = Matrix([[v2w[i][j] for j in range(4)] for i in range(4)])
            cam_obj.matrix_world = M

        cam.clip_start = 0.01
        cam.clip_end   = 10000.0
        return cam_obj, cam

    # 其他情况：参数不对
    raise RuntimeError("无效的投影源：必须是 CAMERA 或 VIEWPORT。")

def _ensure_cycles(scene):
    orig = scene.render.engine
    if orig != 'CYCLES':
        scene.render.engine = 'CYCLES'
    return orig

def _add_uv_project_modifier(obj, uv_name:str, projector_obj, image_aspect=(1.0,1.0)):
    """
    给对象加一个 UV Project 修饰器，把投影结果写入 uv_name。
    注意：Bake 前后我们会删除此修饰器。
    """
    mod = obj.modifiers.new("PSB_UVProject", type='UV_PROJECT')
    mod.uv_layer = uv_name
    # 设置单个投影器
    proj = mod.projectors
    if len(proj) < 1:
        proj.add()
    proj[0].object = projector_obj
    mod.aspect_x = image_aspect[0]
    mod.aspect_y = image_aspect[1]
    # scale/offset 默认 1/0 即可
    return mod

def _build_bake_material(obj, img_src, uvproj_uvmap: str, target_img):
    """
    创建一个临时材质节点树用于烘焙：
      TexCoord(UV=uvproj_uvmap) -> ImageTexture(img_src) -> Emission -> Output
    同时创建一个“Target” Image Texture 节点指向 target_img，并将其设为 active（bake 的落盘目标）。
    返回(材质, 节点树, target_node)
    """
    mat = bpy.data.materials.new(name="PSB_TempBakeMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes: nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    emi = nt.nodes.new("ShaderNodeEmission"); emi.location = (200, 0)
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.location = (0, 0)
    tc  = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-220, 0)
    uvn = nt.nodes.new("ShaderNodeUVMap");   uvn.location = (-220, -150)
    uvn.uv_map = uvproj_uvmap

    tex.image = img_src
    # 使用指定 UVMap
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

    # 目标贴图节点（不连线），设置为 active 以接收 Bake
    target = nt.nodes.new("ShaderNodeTexImage"); target.location = (0, -240)
    target.image = target_img
    nt.nodes.active = target

    # 绑定材质到对象
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    return mat, nt, target

def _bake_emit_to_image(context, obj, target_img, margin_px=4):
    """
    使用 Cycles 执行 Emit 烘焙，结果写入 target_img（取 active image texture）。
    """
    scene = context.scene
    orig_engine = _ensure_cycles(scene)

    # 选择对象
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj

    scene.cycles.samples = max(1, getattr(scene.cycles, "samples", 1))
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = margin_px

    bpy.ops.object.bake(type='EMIT')

    # 还原引擎
    if orig_engine != scene.render.engine:
        scene.render.engine = orig_engine

def _save_image_to(path: str, img: bpy.types.Image):
    ensure_dir(os.path.dirname(path))
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()

class PSB_OT_CreateBlankCanvases(bpy.types.Operator):
    """根据当前导出参数，在输出目录生成两张空白画布（UV 与 3D 投影），并写回到属性里。"""
    bl_idname = "psb.create_canvases"
    bl_label  = "生成两张空白画布"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.psb_exporter
        out_dir = to_abs(p.out_dir) if p.out_dir else os.getcwd()
        ensure_dir(out_dir)

        obj_name = p.target_object.name if getattr(p, "target_object", None) else "object"

        # UV 画布：统一命名
        uv_path = make_outpath(out_dir, "uvcanvas", obj_name, "png")
        create_blank_png(int(p.uv_size), int(p.uv_size), uv_path)

        # 3D 画布尺寸：来自视口或相机的导出尺寸
        if p.render_mode == "VIEWPORT":
            w = int(p.viewport_width); h = int(p.viewport_height)
        else:
            s = context.scene
            w = int(s.render.resolution_x * (s.render.resolution_percentage / 100.0))
            h = int(s.render.resolution_y * (s.render.resolution_percentage / 100.0))

        paint3d_path = make_outpath(out_dir, "paint3d", obj_name, "png")
        # 使用“烘焙背景”颜色作为空白 3D 画布底色
        bg = tuple(getattr(p, "bake_background_color", (1.0, 1.0, 1.0, 0.0)))
        create_blank_png(w, h, paint3d_path, color=bg)

        # 回写绝对路径
        p.paint_uv_path = uv_path
        p.paint_3d_path = paint3d_path

        self.report({'INFO'}, "已生成空白画布（绝对路径已写回）")
        return {'FINISHED'}


class PSB_OT_ApplyPaintUV(bpy.types.Operator):
    bl_idname = "psb.apply_paintuv"
    bl_label  = "应用 UV 绘制 → 贴图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.psb_exporter
        targets = pick_bake_targets(p)
        if not targets:
            self.report({'ERROR'}, "未找到要处理的网格物体（请检查目标类型与目标选择）")
            return {'CANCELLED'}
        if not p.paint_uv_path:
            self.report({'ERROR'}, "未设置 UV 绘制图片路径"); return {'CANCELLED'}

        uv_img_path = to_abs(p.paint_uv_path)
        img_new = load_image_from_path(uv_img_path)
        if not img_new:
            self.report({'ERROR'}, "无法加载 UV 绘制图片"); return {'CANCELLED'}

        overlay = bool(getattr(p, "overlay_enabled", False))
        backup  = bool(getattr(p, "backup_enabled", False))
        out_dir = to_abs(p.out_dir) if p.out_dir else os.getcwd()

        for obj in targets:
            # 没有材质或节点，直接绑定
            if not obj.data.materials:
                assign_image_to_basecolor(obj, img_new); continue
            mat = obj.data.materials[0]
            if not mat or not mat.use_nodes:
                assign_image_to_basecolor(obj, img_new); continue

            nt = mat.node_tree
            bsdf = None
            for n in nt.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    bsdf = n; break

            cur_tex = None
            if bsdf:
                for l in nt.links:
                    if l.to_node == bsdf and l.to_socket.name == "Base Color" and l.from_node.type == 'TEX_IMAGE':
                        cur_tex = l.from_node
                        break

            if not overlay or not cur_tex or not cur_tex.image or not cur_tex.image.filepath_raw:
                # 直接替换
                assign_image_to_basecolor(obj, img_new)
                continue

            # 叠加：备份 -> alpha_over -> 保存回原路径
            dst_img = cur_tex.image
            try:
                if backup:
                    backup_dir = os.path.join(out_dir, "backup_texture")
                    copy_backup(to_abs(dst_img.filepath_raw), backup_dir, "texture", obj.name)
            except Exception as e:
                self.report({'WARNING'}, f"备份旧贴图失败：{obj.name}：{e}")

            try:
                alpha_over(dst_img, img_new)  # 尺寸不一致会抛异常
                dst_img.save()
            except Exception as e:
                assign_image_to_basecolor(obj, img_new)
                self.report({'WARNING'}, f"叠加失败（已改为替换）：{obj.name}：{e}")

        self.report({'INFO'}, f"已应用到 {len(targets)} 个物体（叠加={overlay}）")
        return {'FINISHED'}


class PSB_OT_BakeVisibilityMask(bpy.types.Operator):
    """按 JSON 相机投影，输出每个目标对象在相机视角下的可见性二值遮罩（白=可见，黑=不可见）。"""
    bl_idname = "psb.bake_visibility_mask"
    bl_label  = "保存可见性遮罩"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import math, os, json
        p = context.scene.psb_exporter

        targets = pick_bake_targets(p)
        if not targets:
            self.report({'ERROR'}, "未找到要处理的网格物体（请检查目标类型与目标选择）")
            return {'CANCELLED'}

        out_dir  = to_abs(p.out_dir) if p.out_dir else os.getcwd()
        bake_dir = to_abs(getattr(p, "bake_dir", "")) if getattr(p, "bake_dir", "") else os.path.join(out_dir, "baked")
        ensure_dir(bake_dir)

        meta_path = to_abs(p.meta_path)
        if not meta_path or not os.path.exists(meta_path):
            self.report({'ERROR'}, "未找到 metadata JSON，请在面板指定")
            return {'CANCELLED'}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"读取 JSON 失败：{e}")
            return {'CANCELLED'}

        paint_w = int(meta.get("viewport_params", {}).get("width")  or p.uv_size)
        paint_h = int(meta.get("viewport_params", {}).get("height") or p.uv_size)

        try:
            cam_obj, cam_data, used = _camera_from_json(context, meta, paint_w, paint_h)
        except Exception as e:
            self.report({'ERROR'}, f"重建相机失败：{e}")
            return {'CANCELLED'}

        use_gpu_depth = bool(getattr(p, "use_gpu_depth", getattr(p, "use_gpu", False)))
        if use_gpu_depth:
            _enable_cycles_gpu(context.scene)

        depth_exr_path = make_outpath(bake_dir, "depth", targets[0].name, "exr")
        saved_depth = _render_depth_map(
            context, cam_obj, paint_w, paint_h, depth_exr_path, enable_gpu=use_gpu_depth
        )
        depth_img = load_image_from_path(saved_depth)
        if not depth_img:
            try:
                if cam_data is not None:
                    bpy.data.objects.remove(cam_obj, do_unlink=True)
                    bpy.data.cameras.remove(cam_data, do_unlink=True)
            except Exception:
                pass
            self.report({'ERROR'}, "深度图渲染失败")
            return {'CANCELLED'}
        if hasattr(depth_img, "colorspace_settings"):
            depth_img.colorspace_settings.name = 'Non-Color'
        try:
            depth_img.use_mipmap = False
            depth_img.use_interpolation = False
        except:
            pass

        # 深度容差参数
        eps_rel = float(getattr(p, "depth_epsilon_ratio", 5e-5) or 5e-5)
        eps_min = float(getattr(p, "depth_epsilon_min", 1e-5) or 1e-5)
        eps_max_ratio = float(getattr(p, "depth_epsilon_max_ratio", 0.002) or 0.002)

        use_gpu_bake = bool(getattr(p, "use_gpu_bake", getattr(p, "use_gpu", False)))
        saved_count = 0
        last_mask_path = ""

        try:
            for obj in targets:
                target_w, target_h = get_active_uv_image_size(obj)
                if not target_w:
                    target_w = target_h = int(p.uv_size)

                target_img = bpy.data.images.new(
                    name=f"{obj.name}_VisMask",
                    width=target_w,
                    height=target_h,
                    alpha=False,
                    float_buffer=False
                )
                if hasattr(target_img, "colorspace_settings"):
                    target_img.colorspace_settings.name = 'Non-Color'

                proj_uv = ensure_uv_map(obj, "PSB_ProjectedUV")
                mod = obj.modifiers.new("PSB_UVProject", type='UV_PROJECT')
                mod.uv_layer = proj_uv
                proj = mod.projectors
                if len(proj) < 1:
                    proj.add()
                proj[0].object = cam_obj
                mod.aspect_x = float(paint_w) / max(1.0, float(paint_h))
                mod.aspect_y = 1.0

                orig_mats = [m for m in obj.data.materials]

                mat = bpy.data.materials.new(name="PSB_VisMaskBakeMat")
                mat.use_nodes = True
                nt = mat.node_tree
                for n in list(nt.nodes):
                    nt.nodes.remove(n)

                out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1160, 0)
                emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (960,  0)

                geo = nt.nodes.new("ShaderNodeNewGeometry");        geo.location = (-520, -420)
                pos_xf = nt.nodes.new("ShaderNodeVectorTransform"); pos_xf.location = (-320, -420)
                pos_xf.vector_type = 'POINT'; pos_xf.convert_from = 'OBJECT'; pos_xf.convert_to = 'WORLD'
                nt.links.new(geo.outputs["Position"], pos_xf.inputs["Vector"])
                nrm_xf = nt.nodes.new("ShaderNodeVectorTransform"); nrm_xf.location = (-320, -520)
                nrm_xf.vector_type = 'NORMAL'; nrm_xf.convert_from = 'OBJECT'; nrm_xf.convert_to = 'WORLD'
                nt.links.new(geo.outputs["Normal"], nrm_xf.inputs["Vector"])

                cam_loc = cam_obj.matrix_world.translation
                camX = nt.nodes.new("ShaderNodeValue"); camX.location = (-280, -640); camX.outputs[0].default_value = float(cam_loc.x)
                camY = nt.nodes.new("ShaderNodeValue"); camY.location = (-280, -680); camY.outputs[0].default_value = float(cam_loc.y)
                camZ = nt.nodes.new("ShaderNodeValue"); camZ.location = (-280, -720); camZ.outputs[0].default_value = float(cam_loc.z)
                cmb  = nt.nodes.new("ShaderNodeCombineXYZ");       cmb.location  = (-100, -680)
                nt.links.new(camX.outputs[0], cmb.inputs[0])
                nt.links.new(camY.outputs[0], cmb.inputs[1])
                nt.links.new(camZ.outputs[0], cmb.inputs[2])

                vsub  = nt.nodes.new("ShaderNodeVectorMath"); vsub.location  = (100,   -520); vsub.operation = 'SUBTRACT'
                nt.links.new(cmb.outputs["Vector"], vsub.inputs[0])
                nt.links.new(pos_xf.outputs["Vector"], vsub.inputs[1])
                vnorm = nt.nodes.new("ShaderNodeVectorMath"); vnorm.location = (280, -520); vnorm.operation = 'NORMALIZE'
                nt.links.new(vsub.outputs["Vector"], vnorm.inputs[0])

                vdot = nt.nodes.new("ShaderNodeVectorMath"); vdot.location = (460, -520); vdot.operation = 'DOT_PRODUCT'
                nt.links.new(nrm_xf.outputs["Vector"], vdot.inputs[0])
                nt.links.new(vnorm.outputs["Vector"], vdot.inputs[1])

                # 正面双门
                tdeg = float(getattr(p, "mask_front_threshold", 0.5) or 0.5)
                cos_thresh = math.cos(math.radians(90.0 - tdeg))
                gate_soft = nt.nodes.new("ShaderNodeMath"); gate_soft.location = (640, -520); gate_soft.operation = 'GREATER_THAN'
                gate_soft.inputs[1].default_value = cos_thresh
                nt.links.new(vdot.outputs["Value"], gate_soft.inputs[0])

                gate_strict = nt.nodes.new("ShaderNodeMath"); gate_strict.location = (820, -520); gate_strict.operation = 'GREATER_THAN'
                gate_strict.inputs[1].default_value = 0.0
                nt.links.new(vdot.outputs["Value"], gate_strict.inputs[0])

                face_gate = nt.nodes.new("ShaderNodeMath"); face_gate.location = (1000, -520); face_gate.operation = 'MULTIPLY'
                nt.links.new(gate_soft.outputs["Value"],   face_gate.inputs[0])
                nt.links.new(gate_strict.outputs["Value"], face_gate.inputs[1])

                face_gate_socket = face_gate.outputs["Value"]
                if getattr(p, "mask_invert_facing", False):
                    inv_face = nt.nodes.new("ShaderNodeMath"); inv_face.location = (1180, -560); inv_face.operation = 'SUBTRACT'
                    inv_face.inputs[0].default_value = 1.0
                    nt.links.new(face_gate.outputs["Value"], inv_face.inputs[1])
                    face_gate_socket = inv_face.outputs["Value"]

                depth_tex = nt.nodes.new("ShaderNodeTexImage"); depth_tex.location = (0, -300)
                depth_tex.image = depth_img; depth_tex.interpolation = 'Closest'; depth_tex.projection = 'FLAT'; depth_tex.extension = 'CLIP'
                if hasattr(depth_tex, "color_space"):
                    depth_tex.color_space = 'NONE'
                uvn = nt.nodes.new("ShaderNodeUVMap"); uvn.location = (-220, -150); uvn.uv_map = proj_uv
                nt.links.new(uvn.outputs["UV"], depth_tex.inputs["Vector"])

                dist = nt.nodes.new("ShaderNodeVectorMath"); dist.location = (360, -320); dist.operation = 'LENGTH'
                nt.links.new(vsub.outputs["Vector"], dist.inputs[0])

                # 动态 epsilon（同上）
                bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
                if bbox:
                    import mathutils
                    mins = mathutils.Vector((min(pv.x for pv in bbox), min(pv.y for pv in bbox), min(pv.z for pv in bbox)))
                    maxs = mathutils.Vector((max(pv.x for pv in bbox), max(pv.y for pv in bbox), max(pv.z for pv in bbox)))
                    diag = (maxs - mins).length
                else:
                    diag = 1.0
                eps_max = max(1e-6, eps_max_ratio * diag)

                val_rel = nt.nodes.new("ShaderNodeValue"); val_rel.location = (220, -300); val_rel.outputs[0].default_value = eps_rel
                mul_rel = nt.nodes.new("ShaderNodeMath");  mul_rel.location = (300, -300); mul_rel.operation = 'MULTIPLY'
                nt.links.new(depth_tex.outputs["Color"], mul_rel.inputs[0])
                nt.links.new(val_rel.outputs[0],         mul_rel.inputs[1])

                val_min = nt.nodes.new("ShaderNodeValue"); val_min.location = (420, -260); val_min.outputs[0].default_value = eps_min
                eps_max_v = nt.nodes.new("ShaderNodeValue"); eps_max_v.location = (420, -300); eps_max_v.outputs[0].default_value = eps_max

                eps_lo = nt.nodes.new("ShaderNodeMath"); eps_lo.location = (540, -300); eps_lo.operation = 'MAXIMUM'
                nt.links.new(mul_rel.outputs["Value"], eps_lo.inputs[0])
                nt.links.new(val_min.outputs[0],        eps_lo.inputs[1])

                eps_dyn = nt.nodes.new("ShaderNodeMath"); eps_dyn.location = (720, -300); eps_dyn.operation = 'MINIMUM'
                nt.links.new(eps_lo.outputs["Value"], eps_dyn.inputs[0])
                nt.links.new(eps_max_v.outputs[0],    eps_dyn.inputs[1])

                add_eps = nt.nodes.new("ShaderNodeMath"); add_eps.location = (900, -300); add_eps.operation = 'ADD'
                nt.links.new(depth_tex.outputs["Color"], add_eps.inputs[0])
                nt.links.new(eps_dyn.outputs["Value"],  add_eps.inputs[1])

                depth_ok = nt.nodes.new("ShaderNodeMath"); depth_ok.location = (1080, -300); depth_ok.operation = 'LESS_THAN'
                nt.links.new(dist.outputs["Value"], depth_ok.inputs[0])
                nt.links.new(add_eps.outputs["Value"], depth_ok.inputs[1])

                # UV in [0,1]
                sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = ( -40, -60)
                nt.links.new(uvn.outputs["UV"], sep.inputs["Vector"])
                u_ge_0 = nt.nodes.new("ShaderNodeMath"); u_ge_0.location = (140, -40);  u_ge_0.operation = 'GREATER_THAN'; u_ge_0.inputs[1].default_value = 0.0; nt.links.new(sep.outputs["X"], u_ge_0.inputs[0])
                u_le_1 = nt.nodes.new("ShaderNodeMath"); u_le_1.location = (140, -90);  u_le_1.operation = 'LESS_THAN';   u_le_1.inputs[1].default_value = 1.0; nt.links.new(sep.outputs["X"], u_le_1.inputs[0])
                v_ge_0 = nt.nodes.new("ShaderNodeMath"); v_ge_0.location = (140, -160); v_ge_0.operation = 'GREATER_THAN'; v_ge_0.inputs[1].default_value = 0.0; nt.links.new(sep.outputs["Y"], v_ge_0.inputs[0])
                v_le_1 = nt.nodes.new("ShaderNodeMath"); v_le_1.location = (140, -210); v_le_1.operation = 'LESS_THAN';   v_le_1.inputs[1].default_value = 1.0; nt.links.new(sep.outputs["Y"], v_le_1.inputs[0])
                uv_and1 = nt.nodes.new("ShaderNodeMath"); uv_and1.location = (320, -80);  uv_and1.operation = 'MULTIPLY'; nt.links.new(u_ge_0.outputs["Value"], uv_and1.inputs[0]); nt.links.new(u_le_1.outputs["Value"], uv_and1.inputs[1])
                uv_and2 = nt.nodes.new("ShaderNodeMath"); uv_and2.location = (320, -190); uv_and2.operation = 'MULTIPLY'; nt.links.new(v_ge_0.outputs["Value"], uv_and2.inputs[0]); nt.links.new(v_le_1.outputs["Value"], uv_and2.inputs[1])
                uv_in01 = nt.nodes.new("ShaderNodeMath"); uv_in01.location = (520, -140); uv_in01.operation = 'MULTIPLY'; nt.links.new(uv_and1.outputs["Value"], uv_in01.inputs[0]); nt.links.new(uv_and2.outputs["Value"], uv_in01.inputs[1])

                vis_and = nt.nodes.new("ShaderNodeMath"); vis_and.location = (1260, -360); vis_and.operation = 'MULTIPLY'
                nt.links.new(face_gate_socket,          vis_and.inputs[0])
                vis_and2 = nt.nodes.new("ShaderNodeMath"); vis_and2.location = (1420, -360); vis_and2.operation = 'MULTIPLY'
                nt.links.new(depth_ok.outputs["Value"],  vis_and2.inputs[0])
                nt.links.new(uv_in01.outputs["Value"],   vis_and2.inputs[1])

                mask_final = nt.nodes.new("ShaderNodeMath"); mask_final.location = (1580, -360); mask_final.operation = 'MULTIPLY'
                nt.links.new(vis_and.inputs["Value"],  mask_final.inputs[0])
                nt.links.new(vis_and2.outputs["Value"], mask_final.inputs[1])

                # 严格黑白输出
                mix = nt.nodes.new("ShaderNodeMixRGB"); mix.location = (960,  0); mix.blend_type = 'MIX'
                mix.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)  # 黑
                mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)  # 白
                nt.links.new(mask_final.outputs["Value"], mix.inputs["Fac"])

                nt.links.new(mix.outputs["Color"],  emi.inputs["Color"])
                emi.inputs["Strength"].default_value = 1.0
                nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

                target_node = nt.nodes.new("ShaderNodeTexImage"); target_node.location = (520, -220)
                target_node.image = target_img
                nt.nodes.active = target_node

                if not obj.data.materials: obj.data.materials.append(mat)
                else: obj.data.materials[0] = mat

                scene = context.scene
                orig_engine = scene.render.engine
                try:
                    scene.render.engine = 'CYCLES'
                    if use_gpu_bake:
                        _enable_cycles_gpu(scene)

                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj

                    if hasattr(scene, "cycles"):
                        scene.cycles.samples = 1
                    scene.render.bake.use_clear = True
                    scene.render.bake.use_selected_to_active = False
                    scene.render.bake.margin = 2

                    bpy.ops.object.bake(type='EMIT')

                    mask_path = make_outpath(bake_dir, "vismask", obj.name, "png")
                    target_img.filepath_raw = mask_path
                    target_img.file_format  = 'PNG'
                    target_img.save()

                    last_mask_path = mask_path
                    saved_count += 1

                finally:
                    try: obj.modifiers.remove(mod)
                    except: pass
                    try:
                        obj.data.materials.clear()
                        for m in orig_mats:
                            obj.data.materials.append(m)
                    except: pass
                    try:
                        bpy.data.materials.remove(mat, do_unlink=True)
                    except: pass
                    scene.render.engine = orig_engine

        finally:
            if cam_data is not None:
                try:
                    bpy.data.objects.remove(cam_obj, do_unlink=True)
                except:
                    pass
                try:
                    bpy.data.cameras.remove(cam_data, do_unlink=True)
                except:
                    pass

        if last_mask_path:
            p.mask_path = last_mask_path

        self.report({'INFO'}, f"已保存可见性遮罩：{saved_count} 个对象（模式：{used}）")
        return {'FINISHED'}




# ========= 贴到 apply_ops.py（与其它函数/类并列）=========

def _psb_defaults_dict():
    return {
        # ① 目标/来源/基本导出
        "bake_target_mode": "SINGLE",
        "target_collection": None,
        "render_mode": "CAMERA",
        "camera": None,
        "target_object": None,
        "out_dir": "",
        "uv_size": 4096,
        "viewport_width": 1920,
        "viewport_height": 1080,

        # ② 路径
        "meta_path": "",
        "paint_uv_path": "",
        "paint_3d_path": "",
        "baked_path": "",

        # ③ 遮挡/遮罩
        "mask_visible_color": (0.0, 0.0, 0.0, 1.0),
        "mask_hidden_color": (1.0, 1.0, 1.0, 1.0),
        "mask_front_threshold": 0.5,          # 角度（度）
        "depth_epsilon_ratio": 0.00005,       # 深度相对容差（更小更保守）
        "depth_epsilon_min": 1e-5,            # 深度容差下限
        "depth_epsilon_max_ratio": 0.002,     # 相对对角线的上限比
        "mask_invert_facing": False,          # 默认不反转

        # ④ 烘焙
        "bake_background_color": (1.0, 1.0, 1.0, 0.0),

        # ⑤ 性能
        "use_gpu": True,
        "parallel_workers": 0,
        # "use_gpu_depth": True,
        # "use_gpu_bake": True,
    }


def _reset_psb_props(props):
    """将 props 上与默认表匹配的字段恢复为默认值。"""
    defaults = _psb_defaults_dict()
    for k, v in defaults.items():
        if hasattr(props, k):
            try:
                setattr(props, k, v)
            except Exception:
                # 个别属性（如 PointerProperty）在某些状态下可能拒绝赋值，直接忽略
                pass


class PSB_OT_ResetAllSettings(bpy.types.Operator):
    """重置所有纹理桥设置为默认值"""
    bl_idname = "psb.reset_all_settings"
    bl_label = "重置所有设置"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = getattr(context.scene, "psb_exporter", None)
        if p is None:
            self.report({'ERROR'}, "未找到 PSB 设置（Scene.psb_exporter 不存在）")
            return {'CANCELLED'}

        _reset_psb_props(p)

        # 额外：把面板里显示的最近输出提示清空（若你用它显示结果）
        if hasattr(p, "baked_path"):
            try:
                p.baked_path = ""
            except Exception:
                pass

        self.report({'INFO'}, "已恢复默认设置")
        return {'FINISHED'}




