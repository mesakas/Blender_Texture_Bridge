# __init__.py
bl_info = {
    "name": "Texture Bridge: Camera & UV Exporter",
    "author": "XiaoMing",
    "version": (0, 2, 0),
    "blender": (3, 3, 0),
    "location": "3D Viewport > N 面板 > PS Bridge",
    "description": "Export And Update Texture for Model.",
    "category": "Import-Export",
}

import importlib
import bpy

# --- 子模块导入/热重载 ---
# 注意：必须先 import 子模块，才能在 classes 里引用它们的类
if "properties" in locals():
    importlib.reload(properties)   # type: ignore
else:
    from . import properties

if "utils" in locals():
    importlib.reload(utils)        # type: ignore
else:
    from . import utils

if "export_ops" in locals():
    importlib.reload(export_ops)   # type: ignore
else:
    from . import export_ops

# ★ 关键：确保 apply_ops 被导入（你的新功能就在这里）
if "apply_ops" in locals():
    importlib.reload(apply_ops)    # type: ignore
else:
    from . import apply_ops

if "panel" in locals():
    importlib.reload(panel)        # type: ignore
else:
    from . import panel


# --- 注册表 ---
classes = (
    properties.PSB_ExporterProps,
    export_ops.PSB_OT_Export,
    apply_ops.PSB_OT_CreateBlankCanvases,
    apply_ops.PSB_OT_ApplyPaint3D,
    apply_ops.PSB_OT_ApplyPaintUV,
    apply_ops.PSB_OT_BakeVisibilityMask,
    panel.PSB_PT_Panel,
)



def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.psb_exporter = bpy.props.PointerProperty(type=properties.PSB_ExporterProps)


def unregister():
    del bpy.types.Scene.psb_exporter
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
