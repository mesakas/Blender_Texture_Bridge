# properties.py
import bpy

class PSB_ExporterProps(bpy.types.PropertyGroup):

    bake_target_mode: bpy.props.EnumProperty(
        name="目标类型",
        description="选择要对单个物体还是集合进行处理",
        items=[
            ("SINGLE", "单个物体", "仅处理下方指定的目标物体"),
            ("COLLECTION", "集合", "处理集合中的所有网格物体"),
        ],
        default="SINGLE",
    )

    target_collection: bpy.props.PointerProperty(
        name="目标集合",
        type=bpy.types.Collection,
        description="选择一个集合，对其中所有网格物体进行处理"
    )

    # （其余原有属性保留不变）
    render_mode: bpy.props.EnumProperty(
        name="渲染来源",
        description="选择从相机渲染或从视口抓取",
        items=[
            ("CAMERA",   "相机",   "使用所选相机进行渲染"),
            ("VIEWPORT", "视口",   "抓取第一个 3D 视口进行渲染"),
        ],
        default="CAMERA",
    )


    render_mode: bpy.props.EnumProperty(
        name="渲染来源",
        description="选择从相机渲染或从视口抓取",
        items=[
            ("CAMERA",   "相机",   "使用所选相机进行渲染"),
            ("VIEWPORT", "视口",   "抓取第一个 3D 视口进行渲染"),
        ],
        default="CAMERA",
    )

    meta_path: bpy.props.StringProperty(
        name="Metadata JSON",
        subtype='FILE_PATH',
        default=""
    )

    camera: bpy.props.PointerProperty(
        name="相机",
        type=bpy.types.Object,
        description="用于渲染及导出参数的相机对象",
        poll=lambda self, obj: (obj.type == 'CAMERA')
    )
    target_object: bpy.props.PointerProperty(
        name="目标物体",
        type=bpy.types.Object,
        poll=lambda self, obj: (obj.type == 'MESH')
    )
    out_dir: bpy.props.StringProperty(
        name="输出目录",
        subtype='DIR_PATH',
        default=""
    )
    uv_size: bpy.props.IntProperty(
        name="UV 导出尺寸",
        default=4096,
        min=256,
        soft_max=8192
    )

    # === 视口输出尺寸 ===
    viewport_width: bpy.props.IntProperty(
        name="视口宽度",
        default=1920, min=64, soft_max=8192
    )
    viewport_height: bpy.props.IntProperty(
        name="视口高度",
        default=1080, min=64, soft_max=8192
    )

    # —— 画布与结果路径（相对/绝对均可）——
    paint_uv_path: bpy.props.StringProperty(
        name="UV 绘制 PNG", subtype='FILE_PATH', default=""
    )
    paint_3d_path: bpy.props.StringProperty(
        name="3D 绘制 PNG", subtype='FILE_PATH', default=""
    )
    baked_path: bpy.props.StringProperty(
        name="烘焙输出路径", subtype='FILE_PATH', default=""
    )

    # --- 烘焙输出根目录（默认 out_dir/baked） ---
    bake_dir: bpy.props.StringProperty(
        name="烘焙输出目录",
        description="3D 烘焙、深度EXR与日志的输出目录；为空时自动使用 输出目录/out_dir/baked",
        subtype='DIR_PATH',
        default=""
    )

    # --- 叠加与备份 ---
    overlay_enabled: bpy.props.BoolProperty(
        name="叠加到现有贴图",
        description="将新绘制叠加到当前材质的贴图上（Alpha覆盖）；尺寸需一致，否则改为直接替换",
        default=False
    )
    backup_enabled: bpy.props.BoolProperty(
        name="备份旧贴图",
        description="在覆盖/叠加前，把旧贴图备份到 out_dir/backup_texture 中",
        default=True
    )


    # ===== 两个独立 GPU 开关 =====
    use_gpu_depth: bpy.props.BoolProperty(
        name="深度/遮罩渲染用 GPU（Cycles）",
        description="用于渲染深度图/可见性遮罩",
        default=True,
    )
    use_gpu_bake: bpy.props.BoolProperty(
        name="烘焙贴图用 GPU（Cycles）",
        description="用于烘焙贴图（EMIT）",
        default=True,
    )


    # —— 烘焙背景（放在烘焙按钮前面的那一组里）——
    bake_background_color: bpy.props.FloatVectorProperty(
        name="烘焙背景（RGBA）",
        description="ApplyPaint3D 中未被投射/不可见的像素将使用此颜色与透明度",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 0.0)
    )


    # ====== 可见性/遮挡 控制参数（新增） ======
    mask_visible_color: bpy.props.FloatVectorProperty(
        name="可见颜色",
        description="可见区域的颜色，用于可见性遮罩图",
        subtype='COLOR',
        min=0.0, max=1.0,
        size=4,
        default=(0.0, 0.0, 0.0, 1.0)  # 黑（带 Alpha=1）
    )
    mask_hidden_color: bpy.props.FloatVectorProperty(
        name="不可见颜色",
        description="不可见区域的颜色，用于可见性遮罩图",
        subtype='COLOR',
        min=0.0, max=1.0,
        size=4,
        default=(1.0, 1.0, 1.0, 1.0)  # 白（带 Alpha=1）
    )
    mask_front_threshold: bpy.props.FloatProperty(
        name="正面阈值（度）",
        description="法线与视线夹角小于(90°-阈值)才判为正面，用于抑制掠射角抖动",
        default=0.5,
        min=0.0, soft_max=5.0
    )
    depth_epsilon_ratio: bpy.props.FloatProperty(
        name="深度容差（比例）",
        description="与包围盒对角线相乘得到深度比较的绝对容差，防止Z抖动与插值误差",
        default=0.001,
        min=1e-6, soft_max=0.01
    )
    mask_invert_facing: bpy.props.BoolProperty(
        name="反相遮罩（可见=黑）",
        description="勾选：可见=黑，不可见=白；取消：可见=白，不可见=黑",
        default=True
    )

    # ★ 新增：3D 烘焙未投射区域的背景颜色（含透明度）
    bake_background_color: bpy.props.FloatVectorProperty(
        name="烘焙背景（RGBA）",
        description="ApplyPaint3D 中未被投射/不可见的像素将使用此颜色与透明度",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 0.0)  # 默认白色且完全透明
    )

    # 是否使用 GPU（Cycles）
    use_gpu: bpy.props.BoolProperty(
        name="使用GPU（Cycles）",
        description="启用 Cycles GPU 渲染（需要你的 Blender 首选项中已启用相应显卡设备）",
        default=True,
    )

    # 并行进程数（0=自动=CPU核心数）
    parallel_workers: bpy.props.IntProperty(
        name="并行进程数",
        description="并行烘焙的进程数量；0 表示自动（使用 CPU 逻辑核心数）",
        default=0,
        min=0,
        max=64,
    )