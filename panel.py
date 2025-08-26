import bpy


class PSB_PT_Panel(bpy.types.Panel):
    bl_label = "纹理桥"
    bl_idname = "PSB_PT_PANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "纹理桥"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        p = context.scene.psb_exporter

        # ① 渲染来源
        box = layout.box()
        col = box.column(align=True)
        col.label(text="① 渲染来源", icon="RENDER_STILL")
        row = col.row(align=True)
        row.prop(p, "render_mode", text="来源模式", expand=True)
        if p.render_mode == "VIEWPORT":
            sub = col.column(align=True)
            sub.prop(p, "viewport_width", text="视口宽度 (px)")
            sub.prop(p, "viewport_height", text="视口高度 (px)")
            col.label(text="说明：使用当前 3D 视口的投影视角与分辨率。", icon="INFO")
        else:
            col.prop(p, "camera", text="使用相机")
            col.label(text="说明：按相机参数与姿态进行投影。", icon="INFO")

        # ② 选择目标
        box = layout.box()
        col = box.column(align=True)
        col.label(text="② 选择目标", icon="OUTLINER_OB_GROUP_INSTANCE")
        row = col.row(align=True)
        row.prop(p, "bake_target_mode", text="目标类型", expand=True)
        if p.bake_target_mode == "SINGLE":
            col.prop(p, "target_object", text="单个物体")
        else:
            col.prop(p, "target_collection", text="物体集合")

        # ③ 遮挡/遮罩参数（导出前）
        box = layout.box()
        col = box.column(align=True)
        col.label(text="③ 遮挡/遮罩参数", icon="MOD_MASK")
        col.prop(p, "mask_visible_color", text="可见色")
        col.prop(p, "mask_hidden_color", text="不可见色")
        col.prop(p, "mask_front_threshold", text="正面阈值（度）")
        col.prop(p, "depth_epsilon_ratio", text="深度容差（比例）")
        col.prop(p, "mask_invert_facing", text="反相遮罩（可见=黑）")

        # ④ 输出与绘制（把三个路径字段放在最前）
        box = layout.box()
        col = box.column(align=True)
        col.label(text="④ 输出与绘制", icon="FILE_FOLDER")
        col.prop(p, "out_dir", text="输出目录")
        col.prop(p, "uv_size", text="UV 贴图尺寸 (px)")

        # —— 路径三兄弟（绝对路径将自动写回）——
        col.prop(p, "meta_path", text="Metadata JSON")
        col.prop(p, "paint_uv_path", text="UV 绘制 PNG")
        col.prop(p, "paint_3d_path", text="3D 绘制 PNG")
        col.separator()

        # 导出按钮（更醒目更大）
        row = col.row(align=True); row.scale_y = 1.7
        row.operator("psb.export_camera_uv", text="导出：渲染 + 相机参数 + UV", icon="EXPORT")

        # —— 叠加/备份选项 ——（与贴图应用相关）
        col.separator()
        col.label(text="贴图应用选项", icon="IMAGE_DATA")
        col.prop(p, "overlay_enabled", text="叠加到现有贴图")
        col.prop(p, "backup_enabled", text="覆盖前备份旧贴图（out_dir/backup_texture）")

        # ⑤ 绘制与应用
        box = layout.box()
        col = box.column(align=True)
        col.label(text="⑤ 绘制与应用", icon="BRUSH_DATA")
        # UV 绘制 -> 改名
        row = col.row(align=True)
        row.operator("psb.apply_paintuv", text="将贴图应用到模型", icon="BRUSH_DATA")

        # 3D 烘焙：烘焙目录与背景在按钮之前
        col.prop(p, "bake_dir", text="烘焙输出目录（默认：输出目录/baked）")
        col.prop(p, "bake_background_color", text="烘焙背景（RGBA）")

        row = col.row(align=True); row.scale_y = 1.7
        row.operator("psb.apply_paint3d", text="应用 3D 绘制 → 烘焙贴图", icon="RENDER_STILL")

        # 显示最近输出
        if getattr(p, "baked_path", ""):
            col.separator()
            col.label(text=f"烘焙输出：{p.baked_path}", icon="CHECKMARK")


        # 性能 —— 两个独立 GPU 开关
        box = layout.box()
        col = box.column(align=True)
        col.label(text="性能", icon="PREFERENCES")
        col.prop(p, "use_gpu_depth", text="深度/遮罩渲染用 GPU（Cycles）")
        col.prop(p, "use_gpu_bake", text="烘焙贴图用 GPU（Cycles）")


        # 工具 分组
        box = layout.box()
        col = box.column(align=True)
        col.label(text="工具", icon="TOOL_SETTINGS")
        row = col.row(align=True)
        row.operator("psb.create_canvases", text="生成空白画布（UV + 3D）", icon="ADD")
        row = col.row(align=True)
        row.operator("psb.bake_visibility_mask", text="保存可见性遮罩", icon="IMAGE_ALPHA")
        # 例如在 “工具” box 的末尾：
        row = col.row(align=True)
        row.operator("psb.reset_all_settings", text="重置所有设置", icon="LOOP_BACK")

