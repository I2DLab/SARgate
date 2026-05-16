"""
===============
widgets_size.py
===============

Widget and tab geometry utilities.

Computes all secondary sizes for windows, child windows, tables, plots and
tab-specific elements, applying legacy ratios to the design reference and
the main window geometry previously computed in win_size.get_win_size(state).
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Setup widgets size

from typing import Any


# -----------------------------------------------------------------------------
# 2. Setup widgets size
# -----------------------------------------------------------------------------
def setup_widgets_size(state: dict[str, Any]) -> None:

    vw = int(state["design_ref_width"])
    vh = int(state["design_ref_height"])

    main_win_width  = state["main_win_width"]
    main_win_height = state["main_win_height"]

    win_spacer = int(state["win_spacer"])
    main_win_y = int(state["design_ref_height"] * 0.05)

    state["main_win_y"] = 80


    state["tab_button_size"] = int(state["settings"].get("tab_button_size", 30))
    state["max_tab_button_size"] = 50
    state["tab_button_spacing"] = 3

    state["min_font_scale"] = 0.5
    state["max_font_scale"] = 1.5

    state["file_dialog_width"] = round(vw / 2)
    state["file_dialog_height"] = round(vh / 2)

    state["analysis_win_width"] = round((main_win_width - (win_spacer * 3)) / 4)

    state["analysis_tab_fileselect_height"] = round(main_win_height / 5.57333333)
    state["analysis_init_win_width"] = round(main_win_width / 4.08090909)
    state["analysis_init_file_btn_size"] = round(main_win_width / 25)
    state["analysis_init_load_radius"] = state["analysis_init_file_btn_size"] / 10
    state["analysis_init_chid_input_width"] = round(main_win_width / 10.99236641)
    state["analysis_init_chname_txt_y"] = round(main_win_height / 13.86666667)
    state["analysis_init_chname_input_width"] = round(main_win_width / 10.06993007)

    state["analysis_tab_options_height"] = round(main_win_height / 1.42829828)
    state["analysis_tab_options_y"] = state["analysis_tab_fileselect_height"] + (win_spacer * 2)
    state["analysis_tab_options_spacer"] = round(main_win_height / 41.75)
    state["analysis_tab_options_scaff_width"] = round(main_win_width / 8)
    state["analysis_tab_options_scaff_input_width"] = round(main_win_width / 4.6)
    state["analysis_tab_options_scaff_input_height"] = round(main_win_height / 3)
    state["analysis_tab_options_thresh_width"] = round(main_win_width / 16)
    state["analysis_tab_options_target_width"] = round(main_win_width / 7.2)
    state["analysis_tab_options_target_indent"] = round(main_win_width / 32)
    state["analysis_tab_options_ao_width"] = round(main_win_width / 26.18181818)
    state["analysis_tab_options_ao_indent"] = round(main_win_width / 24.40677966)

    state["analysis_tab_timer_height"] = round(main_win_height / 10.67142857)
    state["analysis_tab_timer_y"] = (
        state["analysis_tab_fileselect_height"]
        + state["analysis_tab_options_height"]
        + (win_spacer * 3)
    )

    state["analysis_win_height"] = main_win_height
    state["analysis_prep_win_x"] = state["analysis_init_win_width"] + (win_spacer * 2)
    state["analysis_scaff_win_x"] = (
        state["analysis_init_win_width"] + state["analysis_win_width"] + (win_spacer * 3)
    )
    state["analysis_dec_win_x"] = (
        state["analysis_init_win_width"] + (state["analysis_win_width"] * 2) + (win_spacer * 4)
    )
    state["analysis_pie_chart_size"] = state["analysis_win_width"]

    state["library_table_width"] = main_win_width
    state["library_table_height"] = main_win_height
    state["library_table_y"] = main_win_y
    state["library_table_img_width"] = round(main_win_width / 5)
    state["library_table_img_height"] = round(main_win_height / 8.35)

    state["overview_img_width"] = round((main_win_width + (win_spacer * 4)) / 3)
    state["overview_img_height"] = round(state["overview_img_width"] * 0.75)
    state["overview_img_win_width"] = main_win_width
    state["overview_img_win_height"] = round(state["overview_img_height"] + (win_spacer * 6))
    state["overview_legend_width"] = round(main_win_width / 12)
    state["overview_legend_height"] = round(main_win_height / 69.58333333)

    state["enrich_plot_win_height"] = round(main_win_height / 5)

    state["overview_dendrogram_width"] = state["overview_img_width"] * 2
    state["overview_dendrogram_x"] = state["overview_img_width"] + (win_spacer * 2)

    state["overview_choose_height"] = round(
        (main_win_height - state["overview_img_win_height"]) - win_spacer
    )
    state["overview_subset_choose_width"] = round(main_win_width / 10.28571429)
    state["overview_choose_width"] = round(main_win_width / 12.30769231)
    state["overview_choose_y"] = state["overview_img_win_height"] + (win_spacer * 2)

    state["overview_prop_x"] = (
        (win_spacer * 4)
        + (state["overview_choose_width"] * 2)
        + state["overview_subset_choose_width"]
    )
    state["overview_prop_width"] = round(
        main_win_width
        - (state["overview_choose_width"] * 2)
        - state["overview_subset_choose_width"]
        - (win_spacer * 3)
    )
    state["overview_prop_child_left_width"] = round(state["overview_prop_width"] * 0.3)
    state["overview_prop_child_right_width"] = round(state["overview_prop_width"] * 0.7)
    state["overview_prop_child_height"] = round(
        state["overview_choose_height"] - (win_spacer * 2)
    )

    state["overview_prop_scaff_smiles_width"] = round(main_win_width / 1.56862745)
    state["overview_prop_child_input_width"] = round(main_win_width / 5.76)
    state["overview_prop_child_button_width"] = round(main_win_width / 9.6)
    state["overview_prop_child_small_button_width"] = round(main_win_width / 57.6)
    state["overview_prop_right_input_width"] = round(main_win_width / 2.25)

    state["similarity_tan_img_width"] = round(main_win_width / 3)
    state["similarity_tan_img_height"] = round(state["similarity_tan_img_width"] * 0.75)
    state["similarity_clustered_img_width"] = round(main_win_width / 4.5)
    state["similarity_clustered_img_height"] = round(
        state["similarity_clustered_img_width"] * 0.75
    )

    state["similarity_manager_win_width"] = round(main_win_width / 3.38028169)
    state["similarity_manager_win_height"] = round(main_win_height / 11.24324324)
    state["similarity_manager_combo_width"] = round(main_win_width / 11.24324324)

    state["similarity_tan_win_height"] = round(
        main_win_height - state["similarity_manager_win_height"] - win_spacer
    )
    state["similarity_tan_win_y"] = (
        state["similarity_manager_win_height"] + (win_spacer * 2)
    )
    state["similarity_tan_mtx_size"] = round(main_win_width / 1.7)
    state["similarity_tan_bar_width"] = round(main_win_width / 72)
    state["similarity_tan_bar_steps"] = round(main_win_height / 0.62839879)

    state["similarity_tbl_win_width"] = main_win_width
    state["similarity_tbl_win_height"] = state["similarity_tan_win_height"]
    state["similarity_tbl_win_y"] = state["similarity_tan_win_y"]
    state["similarity_tbl_img_width"] = round(state["similarity_tbl_win_width"] / 4)

    state["counts_selection_win_width"] = round(main_win_width / 3)
    state["counts_selection_win_height"] = round(main_win_height / 9)
    state["counts_selection_combo_width"] = round(main_win_width / 6.5158371)

    state["counts_info_win_height"] = round(
        main_win_height - state["counts_selection_win_height"] - win_spacer
    )
    state["counts_info_win_y"] = (
        state["counts_selection_win_height"] + (win_spacer * 2)
    )
    state["counts_scaff_img_width"] = round(state["counts_selection_win_width"])
    state["counts_scaff_img_height"] = round(
        state["counts_scaff_img_width"] * 0.75
    )

    state["counts_table_win_width"] = round(
        main_win_width - state["counts_selection_win_width"] - win_spacer
    )
    state["counts_table_win_height"] = main_win_height
    state["counts_table_x"] = (
        state["counts_selection_win_width"] + (win_spacer * 2)
    )
    state["counts_table_img_size"] = round(main_win_width / 6.2)

    state["isomers_manager_win_width"] = (main_win_width - win_spacer) * 0.2
    state["isomers_manager_win_height"] = round(main_win_height / 22.48648649)
    state["isomers_manager_combo_width"] = round(main_win_width / 10)
    state["isomers_manager_combo_spacer"] = win_spacer * 2
    state["isomers_groups_selector_width"] = (main_win_width - win_spacer) * 0.8
    state["isomers_groups_selector_x"] = (
        state["isomers_manager_win_width"] + win_spacer * 2
    )

    state["isomers_images_main_window_width"] = main_win_width
    state["isomers_images_main_window_height"] = round(
        main_win_height - state["isomers_manager_win_height"] - win_spacer
    )
    state["isomers_images_main_window_y"] = (
        state["isomers_manager_win_height"] + (win_spacer * 2)
    )

    state["isomers_table_img_width"] = round(main_win_width / 4)
    state["isomers_table_img_height"] = round(
        state["isomers_table_img_width"] * 0.75
    )

    state["mmpa_img_width"] = round((main_win_width - state["win_spacer"] * 8) / 4)
    state["mmpa_img_height"] = round(state["mmpa_img_width"] * 0.75)
    state["mmpa_img_win_width"] = main_win_width
    state["mmpa_img_win_height"] = state["mmpa_img_height"] + (win_spacer * 2)

    state["mmpa_manager_win_width"] = round(main_win_width / 1.44)
    state["mmpa_manager_win_height"] = round(main_win_height / 12)
    state["mmpa_manager_combo_width"] = round(main_win_width / 7.5)
    state["mmpa_manager_combo_spacer"] = win_spacer * 2

    state["mmpa_table_height"] = round(main_win_height / 1.8225)
    state["mmpa_table_y"] = (
        state["mmpa_manager_win_height"] + (win_spacer * 2)
    )
    state["mmpa_img_win_y"] = (
        state["mmpa_manager_win_height"]
        + state["mmpa_table_height"]
        + (win_spacer * 3)
    )

    state["mmpa_plot_win_width"] = (
        main_win_width - state["mmpa_manager_win_width"] - win_spacer
    )
    state["mmpa_plot_win_height"] = (
        state["mmpa_table_height"] + state["mmpa_manager_win_height"] + win_spacer
    )
    state["mmpa_plot_win_x"] = (
        state["mmpa_manager_win_width"] + (win_spacer * 2)
    )

    ref_w = vw
    state["mmpa_network_map_width"] = round(ref_w / 1.33333333)
    state["mmpa_network_map_height"] = state["mmpa_network_map_width"] * 0.75
    state["mmpa_network_map_x"] = (
        ref_w - state["mmpa_network_map_width"]
    ) / 2

    state["landscape_manager_win_width"] = main_win_width
    state["landscape_manager_win_height"] = round(main_win_height / 12)
    state["landscape_manager_combo_width"] = round(main_win_width / 7.5)
    state["landscape_manager_combo_spacer"] = win_spacer * 2

    state["landscape_plot_win_width"] = main_win_width
    state["landscape_plot_win_height"] = round(main_win_height / 1.115)
    state["landscape_plot_win_y"] = (
        state["landscape_manager_win_height"] + (win_spacer * 2)
    )

    state["landscape_plot_width"] = round(main_win_width / 1.66089965)
    state["landscape_plot_height"] = round(main_win_height / 1.141)

    state["landscape_bar_win_width"] = round(main_win_width / 18)

    state["landscape_img_width"] = round(main_win_width / 3.42857143)
    state["landscape_img_win_width"] = round(main_win_width / 3.27272727)

    state["plots_manager_win_width"] = main_win_width
    state["plots_manager_win_height"] = round(main_win_height / 22.48648649)
    state["plots_manager_combo_width"] = round(main_win_width / 14.4)
    state["plots_manager_combo_spacer"] = win_spacer * 2

    state["plots_main_win_width"] = main_win_width
    state["plots_main_win_height"] = round(main_win_height / 1.07078507)
    state["plots_main_win_y"] = (
        state["plots_manager_win_height"] + (win_spacer * 2)
    )

    state["plots_boxplot_width"] = round(main_win_width / 1.35849057)
    state["plots_boxplot_height"] = round(main_win_height / 1.09473684)
    state["plots_boxplot_det_win_height"] = round(main_win_height / 1.09473684)
    state["plots_boxplot_det_img_size"] = round(main_win_width / 4.5)

    state["plots_heatmap_width"] = round(main_win_width / 1.4229249)
    state["plots_heatmap_height"] = round(main_win_height / 1.09473684)
    state["plots_heatmap_img_width"] = round(main_win_width / 5.53846154)
    state["plots_heatmap_img_height"] = round(
        state["plots_heatmap_img_width"] * 0.75
    )

    state["plots_descriptors_width"] = round(main_win_width / 1.46938776)
    state["plots_descriptors_height"] = round(main_win_height / 1.08051948)
    state["plots_descriptors_img_width"] = round(main_win_width / 3.6)

    state["plots_pca_bar_steps"] = vh * 2
    state["plots_pca_bar_height"] = round(main_win_height / 1.24179104)
    state["plots_pca_bar_win_width"] = round(main_win_width / 18)
    state["plots_pca_bar_win_height"] = round(main_win_height / 1.09473684)
    state["plots_pca_bar_drwlst_height"] = round(main_win_height / 1.17183099)
    state["plots_pca_bar_button_width"] = round(main_win_width / 20.57142857)
    state["plots_pca_bar_xsrt"] = round(main_win_width / 28.8)
    state["plots_pca_bar_xend"] = round(main_win_width / 20.57142857)
    state["plots_pca_bar_ysrt"] = round(main_win_height / 36.17391304)
    state["plots_pca_width"] = round(main_win_width / 1.66089965)
    state["plots_pca_height"] = round(main_win_height / 1.09473684)
    state["plots_pca_img_win_width"] = round(main_win_width / 3.27272727)
    state["plots_pca_img_win_height"] = round(main_win_height / 2.31111111)
    state["plots_pca_clst_win_height"] = round(main_win_height / 2.13333333)
    state["plots_pca_img_width"] = round(main_win_width / 3.42857143)

    state["phph_manager_win_width"] = main_win_width
    state["phph_manager_win_height"] = round(main_win_height / 12.60606061)
    state["phph_manager_combo_width"] = round(main_win_width / 9)
    state["phph_manager_combo_spacer"] = win_spacer * 2
    state["phph_tbl_width"] = round(main_win_width / 1.57033806)
    state["phph_tbl_height"] = round(main_win_height / 1.11229947)
    state["phph_tbl_y"] = (
        state["phph_manager_win_height"] + (win_spacer * 2)
    )

    state["phph_img_width"] = round(main_win_width / 2.94478528)
    state["phph_img_win_width"] = round(main_win_width / 2.8605259)
    state["phph_img_win_height"] = state["phph_tbl_height"]
    state["phph_img_win_x"] = (
        state["phph_tbl_width"] + (win_spacer * 2)
    )

    state["prediction_manager_win_width"] = main_win_width
    state["prediction_manager_win_height"] = round(main_win_height / 12.60606061)
    state["prediction_manager_combo_width"] = round(main_win_width / 6.79245283)
    state["prediction_manager_combo_spacer"] = round(main_win_width / 36)

    state["prediction_tbl_width"] = round(main_win_width / 1.57033806)
    state["prediction_tbl_height"] = round(main_win_height / 1.11229947)
    state["prediction_tbl_y"] = (
        state["prediction_manager_win_height"] + (win_spacer * 2)
    )

    state["prediction_img_width"] = round(main_win_width / 2.94478528)
    state["prediction_img_win_width"] = round(main_win_width / 2.8605259)
    state["prediction_img_win_x"] = (
        state["prediction_tbl_width"] + (win_spacer * 2)
    )
    state["prediction_plot_width"] = round(main_win_width / 2.90909091)
    state["prediction_plot_height"] = round(main_win_height / 2.27945205)

    state["utils_win_width"] = main_win_width
    state["utils_win_height"] = main_win_height

    state["notes_img_width"] = int(round(main_win_width / 1.6))
    state["notes_img_height"] = int(round(state["notes_img_width"] * 0.75))

    state["slith_main_win_width"] = main_win_width
    state["slith_main_win_height"] = main_win_height

    hex_from_width = state["slith_main_win_width"] / (1.5 * (9 - 1) + 10.0)
    hex_from_height = state["slith_main_win_height"] / (1.5 * (9 - 1) + 1.732 + 10.0)
    hex_size = min(hex_from_width, hex_from_height)

    state["slith_hexagon_size"] = round(hex_size)
