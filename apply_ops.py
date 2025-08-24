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
)


import json



def _render_depth_map(context, cam_obj, width, height, out_path):
    """
    使用 Cycles + ViewLayer.material_override 渲染一张“到相机的线性距离图”。
    - 不依赖合成器，不需要打开 Z 通道，稳定性更高
    - 输出 OpenEXR 32-bit（线性），单通道存到 R（或 RGBA 也可以），文件名按 out_path 保存
    返回保存的绝对路径。
    """
    import bpy
    from mathutils import Vector
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
    # 色彩管理（渲染到文件不受视图转换影响，但保持默认更保险）
    cm = scene.view_settings
    orig_view_transform = cm.view_transform
    orig_look = cm.look
    orig_exposure = cm.exposure
    orig_gamma = cm.gamma

    # --- 构造“距离相机”的覆盖材质 ---
    depth_mat = bpy.data.materials.new("PSB_DepthOverrideMat")
    depth_mat.use_nodes = True
    nt = depth_mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (400, 0)

    geo = nt.nodes.new("ShaderNodeNewGeometry");    geo.location = (0, -160)

    # 常量：相机位置
    cam_loc = cam_obj.matrix_world.translation
    camX = nt.nodes.new("ShaderNodeValue"); camX.location = (0, -380); camX.outputs[0].default_value = float(cam_loc.x)
    camY = nt.nodes.new("ShaderNodeValue"); camY.location = (0, -420); camY.outputs[0].default_value = float(cam_loc.y)
    camZ = nt.nodes.new("ShaderNodeValue"); camZ.location = (0, -460); camZ.outputs[0].default_value = float(cam_loc.z)
    cmb  = nt.nodes.new("ShaderNodeCombineXYZ");    cmb.location  = (180, -420)
    nt.links.new(camX.outputs[0], cmb.inputs[0])
    nt.links.new(camY.outputs[0], cmb.inputs[1])
    nt.links.new(camZ.outputs[0], cmb.inputs[2])

    # 视线向量 = CamPos - Position
    vsub = nt.nodes.new("ShaderNodeVectorMath");    vsub.location = (360, -260); vsub.operation = 'SUBTRACT'
    nt.links.new(cmb.outputs["Vector"],   vsub.inputs[0])
    nt.links.new(geo.outputs["Position"], vsub.inputs[1])

    # 距离 = length(CamPos - Position)
    vlen = nt.nodes.new("ShaderNodeVectorMath");    vlen.location = (540, -260); vlen.operation = 'LENGTH'
    nt.links.new(vsub.outputs["Vector"], vlen.inputs[0])

    # 输出到 Emission（灰度），并把同样的值接到 Strength=1，避免额外曝光影响
    nt.links.new(vlen.outputs["Value"], emi.inputs["Color"])
    emi.inputs["Strength"].default_value = 1.0
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

    # --- 应用覆盖材质并渲染 ---
    try:
        scene.camera = cam_obj
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = int(width)
        scene.render.resolution_y = int(height)
        scene.render.resolution_percentage = 100

        # 文件设置
        scene.render.filepath = os.path.splitext(out_path)[0]  # Blender 会自己加扩展名
        img_set.file_format = 'OPEN_EXR'
        img_set.color_mode  = 'BW'        # 单通道
        img_set.color_depth = '32'
        if hasattr(img_set, "exr_codec"):
            img_set.exr_codec = 'ZIP'
        scene.render.film_transparent = True  # 背景透明，背景像素保持 0

        # 关闭任何可能影响数值的显示外观
        cm.view_transform = 'Standard'
        cm.look = 'None'
        cm.exposure = 0.0
        cm.gamma = 1.0

        # 覆盖材质（所有对象都用这一套输出距离）
        view_layer.material_override = depth_mat

        # 渲染并写盘
        bpy.ops.render.render(write_still=True)

        # Blender 实际保存的文件路径（会加 .exr）
        saved = scene.render.filepath
        if not saved.lower().endswith(".exr"):
            saved = saved + ".exr"

    finally:
        # 恢复现场
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

        # 清理材质
        try:
            bpy.data.materials.remove(depth_mat, do_unlink=True)
        except:
            pass

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
    """读取 3D 空间绘制后的图片，依据 metadata.json 的内外参进行精确投影并烘焙到 UV。"""
    bl_idname = "psb.apply_paint3d"
    bl_label  = "应用 3D 绘制 → 贴图（按JSON）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """
        读取 3D 绘制图（paint_3d_path），按 metadata.json 中的相机/视口内外参
        精确重建投影相机，进行“正面 + 深度遮挡”判定后投射到模型并烘焙到 UV。
        仅保存 PNG，不更新 Blender 材质。
        （修复：统一到世界空间、最近邻采样深度、关闭 mipmap、稳定阈值）
        """
        import math

        p   = context.scene.psb_exporter
        obj = p.target_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择网格物体"); return {'CANCELLED'}
        if not p.paint_3d_path:
            self.report({'ERROR'}, "未设置 3D 绘制图片路径"); return {'CANCELLED'}

        # 读取 JSON
        meta_path = bpy.path.abspath(p.meta_path)
        if not meta_path or not os.path.exists(meta_path):
            self.report({'ERROR'}, "未找到 metadata JSON，请在面板指定"); return {'CANCELLED'}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"读取 JSON 失败：{e}"); return {'CANCELLED'}

        # 源图
        src_path = bpy.path.abspath(p.paint_3d_path)
        if not src_path or not os.path.exists(src_path):
            self.report({'ERROR'}, f"找不到 3D 绘制图片：{src_path}"); return {'CANCELLED'}
        img_src = load_image_from_path(src_path)
        if not img_src:
            self.report({'ERROR'}, f"无法加载 3D 绘制图片：{src_path}"); return {'CANCELLED'}
        try: img_src.reload()
        except: pass
        if hasattr(img_src, "colorspace_settings"):
            img_src.colorspace_settings.name = 'sRGB'
        # 关闭 mipmap/interp，避免掠射角颜色渗透
        try:
            img_src.use_mipmap = False
            img_src.use_interpolation = True  # 颜色图保留 Linear 插值
        except: pass
        paint_w, paint_h = int(img_src.size[0]), int(img_src.size[1])

        # 重建相机
        try:
            cam_obj, cam_data, used = _camera_from_json(context, meta, paint_w, paint_h)
        except Exception as e:
            self.report({'ERROR'}, f"重建相机失败：{e}"); return {'CANCELLED'}

        # 渲染深度图（与源图同分辨率）
        out_dir = bpy.path.abspath(p.out_dir); ensure_dir(out_dir)
        depth_exr_path = os.path.join(out_dir, f"{obj.name}_cam_depth.exr")
        saved_depth = _render_depth_map(context, cam_obj, paint_w, paint_h, depth_exr_path)
        depth_img = load_image_from_path(saved_depth)
        if not depth_img:
            self.report({'ERROR'}, "深度图渲染失败"); return {'CANCELLED'}
        if hasattr(depth_img, "colorspace_settings"):
            depth_img.colorspace_settings.name = 'Non-Color'
        try:
            depth_img.use_mipmap = False
            depth_img.use_interpolation = False  # 与节点上的 Closest 保持一致
        except: pass

        # 目标贴图尺寸
        target_w, target_h = get_active_uv_image_size(obj)
        if not target_w:
            target_w = target_h = int(p.uv_size)
        target_img = bpy.data.images.new(name=f"{obj.name}_Baked", width=target_w, height=target_h, alpha=True)

        # UV Project 修饰器（写到临时 UV）
        proj_uv = ensure_uv_map(obj, "PSB_ProjectedUV")
        mod = obj.modifiers.new("PSB_UVProject", type='UV_PROJECT')
        mod.uv_layer = proj_uv
        proj = mod.projectors
        if len(proj) < 1: proj.add()
        proj[0].object = cam_obj
        mod.aspect_x = float(paint_w) / max(1.0, float(paint_h))
        mod.aspect_y = 1.0

        # 备份原材质槽
        orig_mats = [m for m in obj.data.materials]

        # —— 搭建临时材质（正面 + 深度遮挡，全部世界空间）——
        mat = bpy.data.materials.new(name="PSB_TempBakeMat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)

        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1060, 0)
        emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (860, 0)

        # 源图（按临时UV）
        tex = nt.nodes.new("ShaderNodeTexImage");       tex.location = (40, 0)
        tex.image = img_src
        tex.projection = 'FLAT'
        tex.extension  = 'CLIP'
        tex.interpolation = 'Linear'

        uvn = nt.nodes.new("ShaderNodeUVMap");          uvn.location = (-220, -150); uvn.uv_map = proj_uv
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

        # 深度图采样（同一投影视口 UV）
        depth_tex = nt.nodes.new("ShaderNodeTexImage"); depth_tex.location = (40, -300)
        depth_tex.image = depth_img
        depth_tex.interpolation = 'Closest'   # 避免跨像素插值带来的错误
        depth_tex.projection = 'FLAT'
        depth_tex.extension  = 'CLIP'
        if hasattr(depth_tex, "color_space"):
            depth_tex.color_space = 'NONE'
        nt.links.new(uvn.outputs["UV"], depth_tex.inputs["Vector"])

        # 世界空间：Position & Normal
        geo = nt.nodes.new("ShaderNodeNewGeometry");        geo.location = (-460, -420)
        pos_xf = nt.nodes.new("ShaderNodeVectorTransform"); pos_xf.location = (-260, -420)
        pos_xf.vector_type = 'POINT'
        pos_xf.convert_from = 'OBJECT'
        pos_xf.convert_to   = 'WORLD'
        nt.links.new(geo.outputs["Position"], pos_xf.inputs["Vector"])

        nrm_xf = nt.nodes.new("ShaderNodeVectorTransform"); nrm_xf.location = (-260, -520)
        nrm_xf.vector_type = 'NORMAL'
        nrm_xf.convert_from = 'OBJECT'
        nrm_xf.convert_to   = 'WORLD'
        nt.links.new(geo.outputs["Normal"], nrm_xf.inputs["Vector"])

        # 摄像机位置常量（世界）
        cam_loc = cam_obj.matrix_world.translation
        camX = nt.nodes.new("ShaderNodeValue"); camX.location = (-220, -640); camX.outputs[0].default_value = float(cam_loc.x)
        camY = nt.nodes.new("ShaderNodeValue"); camY.location = (-220, -680); camY.outputs[0].default_value = float(cam_loc.y)
        camZ = nt.nodes.new("ShaderNodeValue"); camZ.location = (-220, -720); camZ.outputs[0].default_value = float(cam_loc.z)
        cmb  = nt.nodes.new("ShaderNodeCombineXYZ");       cmb.location  = (-40,  -680)
        nt.links.new(camX.outputs[0], cmb.inputs[0])
        nt.links.new(camY.outputs[0], cmb.inputs[1])
        nt.links.new(camZ.outputs[0], cmb.inputs[2])

        # 视线向量（世界）：normalize(CamPos - WorldPos)
        vsub = nt.nodes.new("ShaderNodeVectorMath");    vsub.location = (160, -520); vsub.operation = 'SUBTRACT'
        nt.links.new(cmb.outputs["Vector"],   vsub.inputs[0])  # CamPos
        nt.links.new(pos_xf.outputs["Vector"], vsub.inputs[1]) # WorldPos

        vnorm = nt.nodes.new("ShaderNodeVectorMath");   vnorm.location = (340, -520); vnorm.operation = 'NORMALIZE'
        nt.links.new(vsub.outputs["Vector"], vnorm.inputs[0])

        # 点乘：dot( WorldNormal, ViewDir )
        vdot = nt.nodes.new("ShaderNodeVectorMath");    vdot.location = (520, -520); vdot.operation = 'DOT_PRODUCT'
        nt.links.new(nrm_xf.outputs["Vector"], vdot.inputs[0])
        nt.links.new(vnorm.outputs["Vector"], vdot.inputs[1])

        # 正面阈值（默认为 0.5°）
        tdeg = float(getattr(p, "mask_front_threshold", 0.5) or 0.5)
        cos_thresh = math.cos(math.radians(90.0 - tdeg))
        step_face = nt.nodes.new("ShaderNodeMath");     step_face.location = (700, -520); step_face.operation = 'GREATER_THAN'
        step_face.inputs[1].default_value = cos_thresh
        nt.links.new(vdot.outputs["Value"], step_face.inputs[0])

        # 距离到相机（世界）
        dist = nt.nodes.new("ShaderNodeVectorMath");    dist.location = (520, -320); dist.operation = 'LENGTH'
        nt.links.new(vsub.outputs["Vector"], dist.inputs[0])

        # 自适应 epsilon
        bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if bbox:
            import mathutils
            mins = mathutils.Vector((min(pv.x for pv in bbox), min(pv.y for pv in bbox), min(pv.z for pv in bbox)))
            maxs = mathutils.Vector((max(pv.x for pv in bbox), max(pv.y for pv in bbox), max(pv.z for pv in bbox)))
            diag = (maxs - mins).length
        else:
            diag = 1.0
        eps_prop = float(getattr(p, "depth_epsilon_ratio", 0.001) or 0.001)
        eps_val = max(1e-6, eps_prop * diag)

        add_eps = nt.nodes.new("ShaderNodeMath");       add_eps.location = (700, -320); add_eps.operation = 'ADD'
        add_eps.inputs[1].default_value = eps_val
        nt.links.new(depth_tex.outputs["Color"], add_eps.inputs[0])

        # 深度测试：dist <= depth + eps
        depth_ok = nt.nodes.new("ShaderNodeMath");      depth_ok.location = (880, -320); depth_ok.operation = 'LESS_THAN'
        nt.links.new(dist.outputs["Value"],    depth_ok.inputs[0])
        nt.links.new(add_eps.outputs["Value"], depth_ok.inputs[1])

        # 同时满足：正面 & 深度
        and_vis = nt.nodes.new("ShaderNodeMath");       and_vis.location = (880, -420); and_vis.operation = 'MULTIPLY'
        nt.links.new(step_face.outputs["Value"], and_vis.inputs[0])
        nt.links.new(depth_ok.outputs["Value"],  and_vis.inputs[1])

        # 可选反相
        mask_socket = and_vis.outputs["Value"]
        if getattr(p, "mask_invert_facing", False):
            inv = nt.nodes.new("ShaderNodeMath");       inv.location = (920, -390); inv.operation = 'SUBTRACT'
            inv.inputs[0].default_value = 1.0
            nt.links.new(and_vis.outputs["Value"], inv.inputs[1])
            mask_socket = inv.outputs["Value"]

        # 用掩码混合：Color = mask * TexColor + (1-mask) * 黑
        mix = nt.nodes.new("ShaderNodeMixRGB");         mix.location = (640, 0)
        mix.blend_type = 'MIX'
        mix.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(mask_socket,           mix.inputs["Fac"])
        nt.links.new(tex.outputs["Color"],  mix.inputs["Color2"])

        nt.links.new(mix.outputs["Color"],  emi.inputs["Color"])
        nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

        # 目标贴图节点（active 接收烘焙）
        target_node = nt.nodes.new("ShaderNodeTexImage"); target_node.location = (260, -240)
        target_node.image = target_img
        nt.nodes.active = target_node

        # 仅为烘焙临时绑定材质到第 0 槽
        if not obj.data.materials: obj.data.materials.append(mat)
        else: obj.data.materials[0] = mat

        # Bake
        scene = context.scene
        orig_engine = scene.render.engine
        try:
            scene.render.engine = 'CYCLES'
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True); context.view_layer.objects.active = obj
            if hasattr(scene, "cycles"): scene.cycles.samples = 1
            scene.render.bake.use_clear = True
            scene.render.bake.use_selected_to_active = False
            scene.render.bake.margin = 2  # 略收紧 margin，减少跨边溢色
            bpy.ops.object.bake(type='EMIT')

            # 保存（不回写材质）
            baked_path = os.path.join(out_dir, f"{obj.name}_baked.png")
            target_img.filepath_raw = baked_path
            target_img.file_format = 'PNG'
            target_img.save()
            p.baked_path = bpy.path.relpath(baked_path)

        finally:
            # 清理
            try: obj.modifiers.remove(mod)
            except: pass
            try:
                obj.data.materials.clear()
                for m in orig_mats:
                    obj.data.materials.append(m)
            except: pass
            if cam_data is not None:
                try: bpy.data.objects.remove(cam_obj, do_unlink=True)
                except: pass
                try: bpy.data.cameras.remove(cam_data, do_unlink=True)
                except: pass
            try: bpy.data.materials.remove(mat, do_unlink=True)
            except: pass
            scene.render.engine = orig_engine

        self.report({'INFO'}, f"已按 JSON 投影并烘焙完成（世界空间+深度遮挡稳定）：{p.baked_path}（模式：{used}）")
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
        out_dir = bpy.path.abspath(p.out_dir)
        ensure_dir(out_dir)

        # UV 画布
        uv_path = os.path.join(out_dir, f"{p.target_object.name}_paint_uv.png")
        create_blank_png(p.uv_size, p.uv_size, uv_path)

        # 3D 画布尺寸：来自视口或相机的导出尺寸
        if p.render_mode == "VIEWPORT":
            w = int(p.viewport_width); h = int(p.viewport_height)
        else:
            s = context.scene
            w = int(s.render.resolution_x * (s.render.resolution_percentage / 100.0))
            h = int(s.render.resolution_y * (s.render.resolution_percentage / 100.0))
        paint3d_path = os.path.join(out_dir, f"{p.target_object.name}_paint_3d.png")
        create_blank_png(w, h, paint3d_path)

        p.paint_uv_path = bpy.path.relpath(uv_path)
        p.paint_3d_path = bpy.path.relpath(paint3d_path)

        self.report({'INFO'}, "已生成空白画布")
        return {'FINISHED'}


class PSB_OT_ApplyPaintUV(bpy.types.Operator):
    """将 UV 绘制图片直接作为模型贴图绑定到材质。"""
    bl_idname = "psb.apply_paintuv"
    bl_label  = "应用 UV 绘制 → 贴图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p   = context.scene.psb_exporter
        obj = p.target_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择网格物体"); return {'CANCELLED'}
        if not p.paint_uv_path:
            self.report({'ERROR'}, "未设置 UV 绘制图片路径"); return {'CANCELLED'}

        img = load_image_from_path(bpy.path.abspath(p.paint_uv_path))
        if not img:
            self.report({'ERROR'}, "无法加载 UV 绘制图片"); return {'CANCELLED'}

        assign_image_to_basecolor(obj, img)
        self.report({'INFO'}, "已将 UV 绘制图片绑定到材质")
        return {'FINISHED'}


class PSB_OT_BakeVisibilityMask(bpy.types.Operator):
    """
    按 metadata.json 重建相机 → UV Project 投影，
    仅烘焙“可见性遮罩”到一张图片（默认：可见=白，不可见=黑）。
    使用 摄像机位置·法线 的点乘 + 阈值判定正面。
    """
    bl_idname = "psb.bake_visibility_mask"
    bl_label  = "保存可见性遮罩"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """
        仅烘焙“可见性遮罩”到一张图片（默认：可见=黑，不可见=白，可配置）。
        可见性的定义：正面判定（世界空间） + 深度可见（通过 Z 深度测试）。
        （修复：统一到世界空间、最近邻采样深度、关闭 mipmap、稳定阈值）
        """
        import math

        p   = context.scene.psb_exporter
        obj = p.target_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择网格物体"); return {'CANCELLED'}

        # 读取 JSON
        meta_path = bpy.path.abspath(p.meta_path)
        if not meta_path or not os.path.exists(meta_path):
            self.report({'ERROR'}, "未找到 metadata JSON，请在面板指定"); return {'CANCELLED'}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"读取 JSON 失败：{e}"); return {'CANCELLED'}

        # 目标贴图尺寸
        target_w, target_h = get_active_uv_image_size(obj)
        if not target_w:
            target_w = target_h = int(p.uv_size)
        target_img = bpy.data.images.new(name=f"{obj.name}_VisMask", width=target_w, height=target_h, alpha=True)

        # 重建相机（深度图尺寸：沿用 metadata 视口尺寸或目标）
        paint_w = int(meta.get("viewport_params",{}).get("width")  or target_w)
        paint_h = int(meta.get("viewport_params",{}).get("height") or target_h)
        try:
            cam_obj, cam_data, used = _camera_from_json(context, meta, paint_w, paint_h)
        except Exception as e:
            self.report({'ERROR'}, f"重建相机失败：{e}"); return {'CANCELLED'}

        # 深度图
        out_dir = bpy.path.abspath(p.out_dir); ensure_dir(out_dir)
        depth_exr_path = os.path.join(out_dir, f"{obj.name}_cam_depth.exr")
        saved_depth = _render_depth_map(context, cam_obj, paint_w, paint_h, depth_exr_path)
        depth_img = load_image_from_path(saved_depth)
        if not depth_img:
            self.report({'ERROR'}, "深度图渲染失败"); return {'CANCELLED'}
        if hasattr(depth_img, "colorspace_settings"):
            depth_img.colorspace_settings.name = 'Non-Color'
        try:
            depth_img.use_mipmap = False
            depth_img.use_interpolation = False
        except: pass

        # UV Project（写入临时 UV）
        proj_uv = ensure_uv_map(obj, "PSB_ProjectedUV")
        mod = obj.modifiers.new("PSB_UVProject", type='UV_PROJECT')
        mod.uv_layer = proj_uv
        proj = mod.projectors
        if len(proj) < 1: proj.add()
        proj[0].object = cam_obj
        mod.aspect_x = float(paint_w) / max(1.0, float(paint_h))
        mod.aspect_y = 1.0

        # 备份原材质槽
        orig_mats = [m for m in obj.data.materials]

        # —— 节点：输出“可见性掩码”（颜色可配置），全部世界空间 —— #
        mat = bpy.data.materials.new(name="PSB_VisMaskBakeMat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)

        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (980, 0)
        emi = nt.nodes.new("ShaderNodeEmission");       emi.location = (780, 0)

        # 世界空间 Position & Normal
        geo = nt.nodes.new("ShaderNodeNewGeometry");        geo.location = (-460, -420)
        pos_xf = nt.nodes.new("ShaderNodeVectorTransform"); pos_xf.location = (-260, -420)
        pos_xf.vector_type = 'POINT'
        pos_xf.convert_from = 'OBJECT'
        pos_xf.convert_to   = 'WORLD'
        nt.links.new(geo.outputs["Position"], pos_xf.inputs["Vector"])

        nrm_xf = nt.nodes.new("ShaderNodeVectorTransform"); nrm_xf.location = (-260, -520)
        nrm_xf.vector_type = 'NORMAL'
        nrm_xf.convert_from = 'OBJECT'
        nrm_xf.convert_to   = 'WORLD'
        nt.links.new(geo.outputs["Normal"], nrm_xf.inputs["Vector"])

        # 摄像机位置（世界）
        cam_loc = cam_obj.matrix_world.translation
        camX = nt.nodes.new("ShaderNodeValue"); camX.location = (-220, -640); camX.outputs[0].default_value = float(cam_loc.x)
        camY = nt.nodes.new("ShaderNodeValue"); camY.location = (-220, -680); camY.outputs[0].default_value = float(cam_loc.y)
        camZ = nt.nodes.new("ShaderNodeValue"); camZ.location = (-220, -720); camZ.outputs[0].default_value = float(cam_loc.z)
        cmb  = nt.nodes.new("ShaderNodeCombineXYZ");       cmb.location  = (-40,  -680)
        nt.links.new(camX.outputs[0], cmb.inputs[0])
        nt.links.new(camY.outputs[0], cmb.inputs[1])
        nt.links.new(camZ.outputs[0], cmb.inputs[2])

        # 视线向量（世界）
        vsub = nt.nodes.new("ShaderNodeVectorMath");    vsub.location = (160, -520); vsub.operation = 'SUBTRACT'
        nt.links.new(cmb.outputs["Vector"],   vsub.inputs[0])
        nt.links.new(pos_xf.outputs["Vector"], vsub.inputs[1])

        vnorm = nt.nodes.new("ShaderNodeVectorMath");   vnorm.location = (340, -520); vnorm.operation = 'NORMALIZE'
        nt.links.new(vsub.outputs["Vector"], vnorm.inputs[0])

        vdot = nt.nodes.new("ShaderNodeVectorMath");    vdot.location = (520, -520); vdot.operation = 'DOT_PRODUCT'
        nt.links.new(nrm_xf.outputs["Vector"], vdot.inputs[0])
        nt.links.new(vnorm.outputs["Vector"], vdot.inputs[1])

        # 正面阈值（默认 0.5°）
        tdeg = float(getattr(p, "mask_front_threshold", 0.5) or 0.5)
        cos_thresh = math.cos(math.radians(90.0 - tdeg))
        step_face = nt.nodes.new("ShaderNodeMath");     step_face.location = (700, -520); step_face.operation = 'GREATER_THAN'
        step_face.inputs[1].default_value = cos_thresh
        nt.links.new(vdot.outputs["Value"], step_face.inputs[0])

        # 深度贴图（最近邻、clip）
        depth_tex = nt.nodes.new("ShaderNodeTexImage"); depth_tex.location = (40, -300)
        depth_tex.image = depth_img
        depth_tex.interpolation = 'Closest'
        depth_tex.projection = 'FLAT'
        depth_tex.extension  = 'CLIP'
        if hasattr(depth_tex, "color_space"):
            depth_tex.color_space = 'NONE'
        uvn = nt.nodes.new("ShaderNodeUVMap");          uvn.location = (-220, -150); uvn.uv_map = proj_uv
        nt.links.new(uvn.outputs["UV"], depth_tex.inputs["Vector"])

        # 自适应 epsilon
        bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if bbox:
            import mathutils
            mins = mathutils.Vector((min(pv.x for pv in bbox), min(pv.y for pv in bbox), min(pv.z for pv in bbox)))
            maxs = mathutils.Vector((max(pv.x for pv in bbox), max(pv.y for pv in bbox), max(pv.z for pv in bbox)))
            diag = (maxs - mins).length
        else:
            diag = 1.0
        eps_prop = float(getattr(p, "depth_epsilon_ratio", 0.001) or 0.001)
        eps_val = max(1e-6, eps_prop * diag)

        # 距离到相机
        dist = nt.nodes.new("ShaderNodeVectorMath");    dist.location = (520, -320); dist.operation = 'LENGTH'
        nt.links.new(vsub.outputs["Vector"], dist.inputs[0])

        add_eps = nt.nodes.new("ShaderNodeMath");       add_eps.location = (700, -320); add_eps.operation = 'ADD'
        add_eps.inputs[1].default_value = eps_val
        nt.links.new(depth_tex.outputs["Color"], add_eps.inputs[0])

        depth_ok = nt.nodes.new("ShaderNodeMath");      depth_ok.location = (880, -320); depth_ok.operation = 'LESS_THAN'
        nt.links.new(dist.outputs["Value"],    depth_ok.inputs[0])
        nt.links.new(add_eps.outputs["Value"], depth_ok.inputs[1])

        and_vis = nt.nodes.new("ShaderNodeMath");       and_vis.location = (880, -420); and_vis.operation = 'MULTIPLY'
        nt.links.new(step_face.outputs["Value"], and_vis.inputs[0])
        nt.links.new(depth_ok.outputs["Value"],  and_vis.inputs[1])

        # 可选反相（默认 True：可见=黑，不可见=白；若你改为 False 则反过来）
        mask_socket = and_vis.outputs["Value"]
        if getattr(p, "mask_invert_facing", True):
            inv = nt.nodes.new("ShaderNodeMath");       inv.location = (920, -390); inv.operation = 'SUBTRACT'
            inv.inputs[0].default_value = 1.0
            nt.links.new(and_vis.outputs["Value"], inv.inputs[1])
            mask_socket = inv.outputs["Value"]

        vis = tuple(getattr(p, "mask_visible_color", (0.0,0.0,0.0,1.0)))
        hid = tuple(getattr(p, "mask_hidden_color", (1.0,1.0,1.0,1.0)))

        mix = nt.nodes.new("ShaderNodeMixRGB");         mix.location = (640, 0)
        mix.blend_type = 'MIX'
        mix.inputs["Color1"].default_value = hid
        mix.inputs["Color2"].default_value = vis
        nt.links.new(mask_socket, mix.inputs["Fac"])

        nt.links.new(mix.outputs["Color"],  emi.inputs["Color"])
        nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])

        # 目标贴图节点（接收烘焙）
        target_node = nt.nodes.new("ShaderNodeTexImage"); target_node.location = (260, -240)
        target_node.image = target_img
        nt.nodes.active = target_node

        # 绑定临时材质
        if not obj.data.materials: obj.data.materials.append(mat)
        else: obj.data.materials[0] = mat

        # Bake
        scene = context.scene
        orig_engine = scene.render.engine
        try:
            scene.render.engine = 'CYCLES'
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True); context.view_layer.objects.active = obj
            if hasattr(scene, "cycles"): scene.cycles.samples = 1
            scene.render.bake.use_clear = True
            scene.render.bake.use_selected_to_active = False
            scene.render.bake.margin = 2
            bpy.ops.object.bake(type='EMIT')

            # 保存
            mask_path = os.path.join(out_dir, f"{obj.name}_visibility_mask.png")
            target_img.filepath_raw = mask_path
            target_img.file_format = 'PNG'
            target_img.save()
            p.mask_path = bpy.path.relpath(mask_path)

        finally:
            try: obj.modifiers.remove(mod)
            except: pass
            try:
                obj.data.materials.clear()
                for m in orig_mats:
                    obj.data.materials.append(m)
            except: pass
            if cam_data is not None:
                try: bpy.data.objects.remove(cam_obj, do_unlink=True)
                except: pass
                try: bpy.data.cameras.remove(cam_data, do_unlink=True)
                except: pass
            try: bpy.data.materials.remove(mat, do_unlink=True)
            except: pass
            scene.render.engine = orig_engine

        self.report({'INFO'}, f"已保存可见性遮罩：{p.mask_path}（模式：{used}）")
        return {'FINISHED'}
