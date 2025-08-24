# panel.py
import bpy

class PSB_PT_Panel(bpy.types.Panel):
    bl_label = "PS Bridge"
    bl_idname = "PSB_PT_PANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PS Bridge"

    def draw(self, context):
        layout = self.layout
        props = context.scene.psb_exporter

        col = layout.column(align=True)
        col.prop(props, "render_mode", text="Render Source")

        if props.render_mode == "VIEWPORT":
            row = col.row(align=True)
            row.prop(props, "viewport_width")
            row.prop(props, "viewport_height")
        else:
            col.prop(props, "camera", text="Camera")

        col.prop(props, "target_object")
        col.prop(props, "out_dir")
        col.prop(props, "uv_size")
        col.prop(props, "meta_path", text="Metadata JSON")
        col.separator()

        col.operator("psb.export_camera_uv", icon="EXPORT")

        # —— 空白画布 & 应用 —— #
        col.separator()
        col.label(text="Painting Canvases")
        col.operator("psb.create_canvases", icon="IMAGE_DATA")

        col.prop(props, "paint_uv_path", text="UV Paint PNG")
        col.operator("psb.apply_paintuv", text="应用 UV 绘制 → 贴图", icon="BRUSH_DATA")

        col.prop(props, "paint_3d_path", text="3D Paint PNG")
        col.operator("psb.apply_paint3d", text="应用 3D 绘制 → 烘焙贴图", icon="RENDER_STILL")
        col.operator("psb.bake_visibility_mask", text="保存可见性遮罩", icon="IMAGE_ALPHA")

        # 颜色与阈值配置
        col.prop(props, "mask_visible_color")
        col.prop(props, "mask_hidden_color")
        col.prop(props, "mask_front_threshold")
        col.prop(props, "mask_invert_facing")
        
        if props.baked_path:
            col.separator()
            col.label(text=f"Baked: {props.baked_path}")
