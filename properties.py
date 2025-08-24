import bpy

class PSB_ExporterProps(bpy.types.PropertyGroup):
    render_mode: bpy.props.EnumProperty(
        name="Render Source",
        description="Choose to render from Camera or capture the active Viewport",
        items=[
            ("CAMERA",   "Camera",   "Render using the selected camera"),
            ("VIEWPORT", "Viewport", "Capture the first 3D Viewport (viewport render)"),
        ],
        default="CAMERA",
    )


    mask_visible_color: bpy.props.FloatVectorProperty(
        name="Mask Visible Color",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),  # 现在默认：可见=黑
        description="可见区域（正面）在遮罩中的颜色"
    )

    mask_hidden_color: bpy.props.FloatVectorProperty(
        name="Mask Hidden Color",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),  # 现在默认：不可见=白
        description="不可见区域（背面）在遮罩中的颜色"
    )

    mask_front_threshold: bpy.props.FloatProperty(
        name="Front Threshold (deg)",
        description="更严格的正面判定：仅当法线与视线夹角 < 90° - 阈值 才视为正面。",
        default=0.0,
        min=0.0, max=60.0,
        soft_max=45.0
    )

    mask_invert_facing: bpy.props.BoolProperty(
        name="Invert Facing",
        description="反相可见性判定（将正面与背面互换）。某些模型/坐标系需要反相。",
        default=True  # ← 按你的情况默认开启
    )



    meta_path: bpy.props.StringProperty(
        name="Metadata JSON",
        subtype='FILE_PATH',
        default=""
    )

    camera: bpy.props.PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Camera to render from (also used to export camera params)",
        poll=lambda self, obj: (obj.type == 'CAMERA')
    )
    target_object: bpy.props.PointerProperty(
        name="Object",
        type=bpy.types.Object,
        poll=lambda self, obj: (obj.type == 'MESH')
    )
    out_dir: bpy.props.StringProperty(
        name="Output Directory",
        subtype='DIR_PATH',
        default=""
    )
    uv_size: bpy.props.IntProperty(
        name="UV Layout Size",
        default=4096,
        min=256,
        soft_max=8192
    )
    mask_path: bpy.props.StringProperty(
        name="Visibility Mask", subtype='FILE_PATH', default=""
    )

    # === 视口输出尺寸 ===
    viewport_width: bpy.props.IntProperty(
        name="Viewport Width",
        default=1920, min=64, soft_max=8192
    )
    viewport_height: bpy.props.IntProperty(
        name="Viewport Height",
        default=1080, min=64, soft_max=8192
    )


    # —— 画布与结果路径（相对/绝对均可）——
    paint_uv_path: bpy.props.StringProperty(
        name="UV Paint Image", subtype='FILE_PATH', default=""
    )
    paint_3d_path: bpy.props.StringProperty(
        name="3D Paint Image", subtype='FILE_PATH', default=""
    )
    baked_path: bpy.props.StringProperty(
        name="Baked Output", subtype='FILE_PATH', default=""
    )
