# utils.py
import os
import time
from mathutils import Matrix, Vector


# utils.py 末尾附近新增

def pick_bake_targets(props):
    """
    根据 props.bake_target_mode 返回要处理的网格对象列表。
    - SINGLE: 返回 [target_object]（若有效）
    - COLLECTION: 返回集合中所有 type=='MESH' 的对象（去重）
    """
    targets = []
    mode = getattr(props, "bake_target_mode", "SINGLE")
    if mode == "COLLECTION":
        col = getattr(props, "target_collection", None)
        if col:
            for o in col.objects:
                if getattr(o, "type", "") == 'MESH':
                    targets.append(o)
    else:
        obj = getattr(props, "target_object", None)
        if obj and getattr(obj, "type", "") == 'MESH':
            targets.append(obj)
    # 去重（同一个数据块可能被多处引用）
    uniq = []
    seen = set()
    for o in targets:
        if o.name not in seen:
            uniq.append(o)
            seen.add(o.name)
    return uniq


def create_blank_png(width:int, height:int, path:str, color=(0,0,0,0)):
    """
    创建一张空 PNG（透明）到磁盘，不在 .blend 中常驻。
    """
    import bpy
    ensure_dir(os.path.dirname(path))
    img = bpy.data.images.new("PSB_Blank", width=width, height=height, alpha=True, float_buffer=False)
    # 填充颜色
    r,g,b,a = color
    px = [r, g, b, a] * (width*height)
    img.pixels.foreach_set(px)
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    try: bpy.data.images.remove(img)
    except: pass

def load_image_from_path(path:str):
    import bpy, os
    if not path or not os.path.exists(path):
        return None
    try:
        return bpy.data.images.load(path, check_existing=True)
    except:
        # 再尝试从已加载列表中找
        for img in bpy.data.images:
            if bpy.path.abspath(img.filepath_raw) == bpy.path.abspath(path):
                return img
        return None

def _find_principled_bsdf(mat):
    if not mat or not mat.use_nodes: return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None

def assign_image_to_basecolor(obj, image):
    """
    将 image 绑定到对象第一个材质的 BaseColor。
    - 若已有 Image Texture -> 替换其 image
    - 否则新建 Image Texture 节点并连接到 Base Color
    """
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="PSB_Mat")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    mat = obj.data.materials[0]
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _find_principled_bsdf(mat)
    if not bsdf:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs["BSDF"], nt.nodes["Material Output"].inputs["Surface"])
    # 查找已有与 BaseColor 相连的 Image Texture
    tex_node = None
    for n in nt.nodes:
        if n.type == 'TEX_IMAGE':
            # 看看是否连到了 Base Color
            for l in nt.links:
                if l.from_node == n and l.to_node == bsdf and l.to_socket.name == "Base Color":
                    tex_node = n; break
    if not tex_node:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (bsdf.location.x - 300, bsdf.location.y)
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    tex_node.image = image

def ensure_uv_map(obj, name:str):
    """
    确保 obj 存在某个 UVMap 名称。若不存在则创建并返回其名称；存在则直接返回。
    """
    me = obj.data
    if name in [uv.name for uv in me.uv_layers]:
        return name
    uv = me.uv_layers.new(name=name, do_init=False)
    return uv.name

def get_active_uv_image_size(obj):
    """
    尝试获取材质里当前用于 BaseColor 的贴图尺寸。若找不到则返回(None,None)。
    """
    if not obj.data.materials: return (None, None)
    mat = obj.data.materials[0]
    if not mat.use_nodes: return (None, None)
    nt = mat.node_tree
    bsdf = None
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            bsdf = n; break
    if not bsdf: return (None, None)
    for l in nt.links:
        if l.to_node == bsdf and l.to_socket.name == "Base Color" and l.from_node.type == 'TEX_IMAGE':
            img = l.from_node.image
            if img: return img.size[0], img.size[1]
    return (None, None)



def compute_projection_matrix_gl(scene, cam_obj):
    """
    基于 compute_camera_K 得到的 K，计算 OpenGL 风格的 4x4 投影矩阵。
    - PERSP: 右手系、相机朝 -Z
    - ORTHO: 标准正交投影
    - PANO/鱼眼: 返回 None
    """
    cam = cam_obj.data
    K, params = compute_camera_K(scene, cam_obj)
    if K is None:
        # 全景/鱼眼等非针孔相机：不生成投影矩阵
        return None

    width  = float(params["resolution_x"])
    height = float(params["resolution_y"])
    n = float(cam.clip_start)
    f = float(cam.clip_end)

    fx = float(K[0][0]); fy = float(K[1][1])
    cx = float(K[0][2]); cy = float(K[1][2])

    if cam.type == 'PERSP':
        # OpenGL 风格透视矩阵（把像素系 K 映射到 NDC）
        a = (2.0 * fx) / width
        b = (2.0 * fy) / height
        c = 1.0 - (2.0 * cx) / width
        d = (2.0 * cy) / height - 1.0

        # 典型 OpenGL 深度项（相机看向 -Z）
        z3 = -(f + n) / (f - n)
        z4 = -(2.0 * f * n) / (f - n)

        P = Matrix((
            (a, 0.0, c, 0.0),
            (0.0, b, d, 0.0),
            (0.0, 0.0, z3, z4),
            (0.0, 0.0, -1.0, 0.0),
        ))
        return [[float(v) for v in row] for row in P]

    elif cam.type == 'ORTHO':
        # 由 compute_camera_K 推得 fx, fy => 每世界单位对应多少像素
        # 屏幕覆盖的世界宽高（在正交投影中与 near 无关）
        w_world = width  / fx
        h_world = height / fy

        l = -0.5 * w_world; r = 0.5 * w_world
        b = -0.5 * h_world; t = 0.5 * h_world

        P = Matrix((
            (2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)),
            (0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)),
            (0.0, 0.0, -2.0 / (f - n), -(f + n) / (f - n)),
            (0.0, 0.0, 0.0, 1.0),
        ))
        return [[float(v) for v in row] for row in P]

    else:
        return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def mat4_to_list(m: Matrix):
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]

def mat3_to_list(m: Matrix):
    return [[float(m[i][j]) for j in range(3)] for i in range(3)]

def vec3_to_list(v: Vector):
    return [float(v[0]), float(v[1]), float(v[2])]

def compute_camera_K(scene, cam_obj):
    """Compute pinhole intrinsics K for PERSP/ORTHO. For PANO return (None, params)."""
    cam = cam_obj.data
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.0
    px_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    width = res_x * scale
    height = res_y * scale

    params = {
        "resolution_x": int(width),
        "resolution_y": int(height),
        "pixel_aspect_xy": [float(scene.render.pixel_aspect_x), float(scene.render.pixel_aspect_y)],
        "sensor_fit": cam.sensor_fit,          # 'AUTO','HORIZONTAL','VERTICAL'
        "sensor_width_mm": float(cam.sensor_width),
        "sensor_height_mm": float(cam.sensor_height),
        "lens_mm": float(cam.lens),
        "type": cam.type,                      # 'PERSP','ORTHO','PANO'
        "lens_unit": getattr(cam, "lens_unit", "MILLIMETERS"),
    }

    if cam.type == 'PERSP':
        sensor_width = cam.sensor_width
        sensor_height = cam.sensor_height
        if cam.sensor_fit == 'VERTICAL':
            fy = (height) * (cam.lens / sensor_height)
            fx = fy / px_aspect
        else:
            fx = (width) * (cam.lens / sensor_width)
            fy = fx * px_aspect

        cx = width * 0.5
        cy = height * 0.5
        K = Matrix(((fx, 0.0, cx),
                    (0.0, fy, cy),
                    (0.0, 0.0, 1.0)))
        return [[float(v) for v in row] for row in K], params

    elif cam.type == 'ORTHO':
        scale_world = cam.ortho_scale
        fx = (width) / scale_world
        fy = (height) / scale_world
        cx = width * 0.5
        cy = height * 0.5
        K = Matrix(((fx, 0.0, cx),
                    (0.0, fy, cy),
                    (0.0, 0.0, 1.0)))
        return [[float(v) for v in row] for row in K], params

    else:
        return None, params

def get_extrinsics(cam_obj):
    """Return camera-to-world/world-to-camera matrices and (R,t) such that X_cam = R * X_world + t."""
    cam_to_world = cam_obj.matrix_world.copy()
    world_to_cam = cam_to_world.inverted()
    R = world_to_cam.to_3x3()
    t = world_to_cam.to_translation()
    return cam_to_world, world_to_cam, R, t



def find_first_view3d(context):
    """在所有窗口中寻找第一个 3D 视口，返回 (window, area, region, space)。找不到则返回 (None, None, None, None)。"""
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
                    return w, a, reg_win, s
    return None, None, None, None

def projection_to_intrinsics_from_gl(P, width, height):
    """
    从 OpenGL 风格投影矩阵 P（4x4）反推出像素系内参 K：
    a = 2*fx/width; b = 2*fy/height; c = 1 - 2*cx/width; d = 2*cy/height - 1
    => fx = a*width/2; fy = b*height/2; cx = (1-c)*width/2; cy = (d+1)*height/2
    """
    a = float(P[0][0])
    b = float(P[1][1])
    c = float(P[0][2])
    d = float(P[1][2])
    fx = a * width * 0.5
    fy = b * height * 0.5
    cx = (1.0 - c) * width * 0.5
    cy = (d + 1.0) * height * 0.5
    K = Matrix(((fx, 0.0, cx),
                (0.0, fy, cy),
                (0.0, 0.0, 1.0)))
    return K

def compute_viewport_calibration(context, width, height):
    """
    计算当前“第一个 3D 视口”的外参/投影矩阵/内参。
    返回 dict:
      {
        "projection_matrix_4x4": list(4x4),
        "world_to_view": list(4x4),
        "view_to_world": list(4x4),
        "R_world_to_view": list(3x3),
        "t_world_to_view": list(3),
        "intrinsics_K": list(3x3),
        "view_perspective": 'PERSP'/'ORTHO'/'CAMERA',
        "lens_mm": float,
      }
    """
    win, area, region, space = find_first_view3d(context)
    if not (win and area and region and space):
        raise RuntimeError("未找到 3D 视口窗口，请确保至少打开一个 3D Viewport。")

    rv3d = space.region_3d
    # Blender 定义：pv = P * V（世界到 NDC 的整体矩阵）
    PV = rv3d.perspective_matrix.copy()
    V = rv3d.view_matrix.copy()
    try:
        P = PV @ V.inverted()
    except:
        # 极少数情况下 V 不可逆（不太可能），保守返回 PV 当作 P（会使 K 不准）
        P = PV.copy()

    # 内参（基于像素宽高）
    K = projection_to_intrinsics_from_gl(P, float(width), float(height))

    data = {
        "projection_matrix_4x4": [[float(P[i][j]) for j in range(4)] for i in range(4)],
        "world_to_view": [[float(V[i][j]) for j in range(4)] for i in range(4)],
        "view_to_world": [[float(V.inverted()[i][j]) for j in range(4)] for i in range(4)],
        "R_world_to_view": [[float(x) for x in V.to_3x3()[i]] for i in range(3)],
        "t_world_to_view": [float(v) for v in V.to_translation()],
        "intrinsics_K": [[float(K[i][j]) for j in range(3)] for i in range(3)],
        "view_perspective": rv3d.view_perspective,  # 'PERSP','ORTHO','CAMERA'
        "lens_mm": getattr(space, "lens", None),
    }
    return data