# panel.py
import bpy

class PSB_PT_Panel(bpy.types.Panel):
    bl_label = "纹理桥"
    bl_idname = "PSB_PT_PANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "纹理桥"

    def draw(self, context):
        layout = self.layout
        props = context.scene.psb_exporter

        col = layout.column(align=True)
        col.prop(props, "render_mode", text="渲染来源")

        if props.render_mode == "VIEWPORT":
            row = col.row(align=True)
            row.prop(props, "viewport_width", text="视口宽度")
            row.prop(props, "viewport_height", text="视口高度")
        else:
            col.prop(props, "camera", text="相机")

        # === 新增：目标类型选择 + 动态输入 ===
        col.separator()
        col.prop(props, "bake_target_mode", text="目标类型")
        if props.bake_target_mode == "SINGLE":
            col.prop(props, "target_object", text="目标物体")
        else:
            col.prop(props, "target_collection", text="目标集合")

        col.prop(props, "out_dir", text="输出目录")
        col.prop(props, "uv_size", text="UV 导出尺寸")
        col.prop(props, "meta_path", text="Metadata JSON 路径")
        col.separator()

        col.operator("psb.export_camera_uv", text="导出（渲染 + 相机参数 + UV）", icon="EXPORT")

        # —— 空白画布 & 应用 —— #
        col.separator()
        col.label(text="绘制画布")
        col.operator("psb.create_canvases", text="生成两张空白画布", icon="IMAGE_DATA")

        col.prop(props, "paint_uv_path", text="UV 绘制 PNG")
        col.operator("psb.apply_paintuv", text="应用 UV 绘制 → 贴图", icon="BRUSH_DATA")

        col.prop(props, "paint_3d_path", text="3D 绘制 PNG")
        col.operator("psb.apply_paint3d", text="应用 3D 绘制 → 烘焙贴图", icon="RENDER_STILL")
        col.operator("psb.bake_visibility_mask", text="保存可见性遮罩", icon="IMAGE_ALPHA")

        col.separator()
        col.label(text="遮挡与遮罩参数")
        col.prop(props, "mask_visible_color", text="可见颜色")
        col.prop(props, "mask_hidden_color", text="不可见颜色")
        col.prop(props, "mask_front_threshold", text="正面阈值（度）")
        col.prop(props, "depth_epsilon_ratio", text="深度容差（比例）")
        col.prop(props, "mask_invert_facing", text="反相遮罩（可见=黑）")
        
        col.prop(props, "bake_background_color", text="烘焙背景（RGBA）")

        if props.baked_path:
            col.separator()
            col.label(text=f"烘焙输出：{props.baked_path}")
