#!/usr/bin/env python3
"""
植物病蟲害辨識 Streamlit Web 應用
基於 ConvNeXt Large 深度學習模型
"""

import streamlit as st
from PIL import Image
import pandas as pd
from predict import PlantDiseasePredictor
import altair as alt



# ====== 自訂背景顏色 + 自製頂部頁首 ======
st.markdown("""
    <style>
        /* 整個背景 */
        .stApp {
            background-color: #768f5f;
        }

        /* 把原本的 Streamlit header 壓扁、變透明 */
        [data-testid="stHeader"] {
            background: transparent;
            height: 0px;
        }


        /* ⭐ 左側 sidebar 背景顏色 */
        [data-testid="stSidebar"] {
            background-color: #52663f;   /* 這裡改成你想要的顏色 */
        }

        /* 側邊欄 expander 標題底色（模型狀態 / 檢視所有類別）*/
        [data-testid="stSidebar"] [data-testid="stExpander"] > details > summary {
            background-color: #3b4f32;   /* 這裡換你喜歡的色 */
            color: #ffffff !important;   /* 標題文字顏色 */
            border-radius: 6px;
        }

        /* 如果不想要 expander 外框的線，就留著；想保留原本外框就刪掉這段 */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: none;
        }

        /* 外層頂部 bar：佔滿整個寬度 */
        .custom-top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 3rem;
            background-color: #768f5f;
            display: flex;
            align-items: center;
            z-index: 999;
        }

        /* 滑桿底線的顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
            background-color: #000000;   
        }

        /* 已填滿的那一段線（左側有值的部分） */
        [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
            background-color: #3b4f32;   
        }

        /* 滑桿圓形手把的顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background-color: #3b4f32;   
            border-color: #3b4f32;       
        }

        /* Slider 上方/下方顯示的數字與文字顏色 */
        [data-testid="stSidebar"] [data-baseweb="slider"] * {
            color: #FFFFFF !important;  /* 換成你要的顏色 */
        }

        /* 修改 expander 標題（上方 summary）背景色 */
        details > summary {
            background-color: #52663f !important;    /* <<< expander 標題底色 */
            color: white !important;
            border-radius: 10px !important;
        }
        /* 調整 st.metric 裡 delta 文字顏色 */
        [data-testid="stMetricDelta"] > div {
            color: #3b4f32 !important;   /* 這裡換成你想要的顏色 */
            font-weight:550;
        }
         /* 改變上升箭頭顏色（避免留著預設亮綠） */
        [data-testid="stMetricDelta"] svg {
            fill: #3b4f32 !important;
            color: #3b4f32 !important;
        }

    </style>
""", unsafe_allow_html=True)




# ========== 載入模型 (快取) ==========
@st.cache_resource
def load_predictor():
    """載入預測器 (只執行一次)"""
    return PlantDiseasePredictor(
        model_path='output_v2/best_model.pth',
        classes_path='output_v2/classes.json',
        verbose=False
    )

try:
    predictor = load_predictor()
    model_info = predictor.get_model_info()
except Exception as e:
    st.error(f"無法載入模型: {e}")
    st.info("請確保 output/best_model.pth 和 output/classes.json 存在")
    st.stop()

# ========== 側邊欄 ==========
with st.sidebar:
    st.header("系統資訊")

    # 模型狀態（改成下拉選單）
    with st.expander("模型狀態", expanded=False):  # expanded=True 代表預設展開
        st.write(f"**類別數量**: {model_info['num_classes']}")
        st.write(f"**計算裝置**: {model_info['device']}")
        if model_info['accuracy']:
            st.write(f"**模型準確率**: {model_info['accuracy']:.2f}%")

    with st.expander("檢視所有類別"):
        for i, cls in enumerate(model_info['class_names'], 1):
            st.write(f"{i}. {cls}")

    st.markdown("---")

    # 預測參數
    st.subheader("預測設定")
    top_k = st.slider(
        "顯示前 K 個結果",
        min_value=1,
        max_value=model_info['num_classes'],
        value=3
    )

    confidence_threshold = st.slider(
        "信心度閾值 (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="低於此閾值會顯示警告"
    )

     # 🎨 在側邊欄最下方放插圖
    st.markdown("---")
    st.image("spy.PNG", use_container_width=True)



# ========== 檔案上傳 ==========
# 上傳元件本身 label 留空，就不會再顯示預設字
uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png'],
    help="請上傳清晰的植物葉片照片以獲得最佳診斷結果"
)

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)

    # 建立兩欄布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上傳的圖片")
        st.image(image, use_container_width=True, caption=uploaded_file.name)

        # 圖片資訊
        with st.expander("檢視圖片資訊"):
            st.write(f"**檔案名稱**: {uploaded_file.name}")
            st.write(f"**圖片尺寸**: {image.size[0]} x {image.size[1]} px")
            st.write(f"**圖片格式**: {image.format}")
            st.write(f"**色彩模式**: {image.mode}")

    with col2:
        st.subheader("診斷結果")

        # 進行預測
        with st.spinner('AI 正在分析圖片...'):
            predictions = predictor.predict(image, top_k=top_k)

        # 最佳預測結果
        best_class, best_prob = predictions[0]

        # 根據信心度顯示不同訊息
        if best_prob >= confidence_threshold:
            result_bg = "#52663f"   
            result_title = "診斷結果"
        else:
            result_bg = "#52663f"   
            result_title = "可能診斷（信心度較低）"

        st.markdown(
        f"""
        <div style="
            background-color:{result_bg};
            border-radius:10px;
            padding:0.8rem 1.0rem;
            color:#ffffff;
            font-weight:600;
            font-size:1.05rem;
            margin-bottom:0.8rem;
        ">
            {result_title}：{best_class}
        </div>
        """,
        unsafe_allow_html=True,
    )


        # 顯示信心度
        st.metric(
            label="診斷信心度",
            value=f"{best_prob:.2f}%",
            delta=f"{best_prob - confidence_threshold:.2f}% vs 閾值"
        )

        # 建議措施
        st.markdown("---")
        st.markdown("### 建議措施")

        disease_recommendations = {
            "healthy": "葉片健康，繼續保持良好的栽培管理。",
            "canker": "檢測到潰瘍病，建議：\n- 移除受感染組織\n- 使用銅基殺菌劑\n- 改善通風條件",
            "greasy_spot": "檢測到油斑病，建議：\n- 噴灑適當殺菌劑\n- 避免過度灌溉與葉面長期潮濕\n- 清除嚴重受害落葉",
            "melanose": "檢測到黑點病，建議：\n- 使用保護性殺菌劑\n- 修剪過密枝條\n- 注意排水與通風",
            "sooty_mold": "檢測到煤煙病，建議：\n- 先控制蚜蟲、介殼蟲等分泌蜜露的害蟲\n- 視情況清洗葉面\n- 改善園區通風與採光",
            "pest_aphid": "檢測到蚜蟲危害，建議：\n- 針對嫩梢與葉背進行防治\n- 可使用皂素、礦物油或選擇性殺蟲劑\n- 避免氮肥過量以減少嫩梢暴露",
            "pest_leaf_miner": "檢測到潛葉蛾危害，建議：\n- 剪除嚴重受害葉片\n- 適時使用系統性殺蟲劑\n- 監測成蟲發生期以提早防治",
            "pest_scale_insect": "檢測到介殼蟲危害，建議：\n- 修剪嚴重受害枝條\n- 使用礦物油或合適殺蟲劑\n- 搭配天敵保育降低族群密度",
            "pest_thrips": "檢測到薊馬危害，建議：\n- 加強花期與嫩葉期監測\n- 適時使用選擇性殺蟲劑\n- 搭配黃色/藍色黏蟲板監控族群變化",
        }

         # 針對不同疾病給不同底色
        disease_colors = {
            "healthy": ("#52663f", "#ffffff"),   # (背景色, 文字色)
            "canker": ("#52663f", "#ffffff"),
            "greasy_spot": ("#52663f", "#ffffff"),
            "melanose": ("#52663f", "#ffffff"),
            "sooty_mold": ("#52663f", "#ffffff"),
            "pest_aphid": ("#52663f", "#ffffff"),
            "pest_leaf_miner": ("#52663f", "#ffffff"),
            "pest_scale_insect": ("#52663f", "#ffffff"),
            "pest_thrips": ("#52663f", "#ffffff"),
        }

        recommendation = disease_recommendations.get(
            best_class,
            "請諮詢專業植物病理學家以獲得詳細建議。"
        )
        bg_color, text_color = disease_colors.get(best_class, ("#52663f", "#ffffff"))
        
        # 用自訂色塊顯示建議內容（保留換行）
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                color:{text_color};
                border-radius:10px;
                padding:0.8rem 1.0rem;
                white-space:pre-line;
                font-size:0.93rem;    
            ">{recommendation}</div>""",
            unsafe_allow_html=True,
        )

# ========== 詳細分析 ==========
    st.markdown("---")
    st.subheader("詳細分析")

    # 建立 DataFrame
    df = pd.DataFrame(predictions, columns=['類別', '信心度 (%)'])
    df['排名'] = range(1, len(df) + 1)
    df = df[['排名', '類別', '信心度 (%)']]

    # --------- 表格：整體顏色風格 ---------
    styled_df = (
        df.style
        # 標題列樣式
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#3b4f32"),  # 標題列底色
                    ("color", "#ffffff"),             # 標題文字顏色
                    ("font-weight", "600"),
                    ("text-align", "center"),
                ],
            }
        ])
        # 資料列樣式
        .set_properties(**{
            "background-color": "#52663f",  # 每一列底色
            "color": "#ffffff",             # 每一列文字顏色
            "border-color": "#768f5f",
        })
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------- 長條圖：整體顏色風格（改用 Altair） ---------
    import altair as alt

    chart = (
        alt.Chart(df)
        .mark_bar(color="#3b4f32")
        .encode(
            x=alt.X(
                "類別:N",
                sort="-y",
                axis=alt.Axis(
                    title=None,
                    labelAngle=0,
                    labelFontSize=14,   # ← x 軸文字大小
                ),
            ),
            y=alt.Y(
                "信心度 (%):Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title=None),
            ),
        )
        .properties(
            height=260,
            width=600,              
            background="#52663f",
        )
        .configure_view(
            strokeWidth=0,
        )
        .configure_axis(
            grid=True,
            gridColor="#768f5f",
            gridOpacity=0.6,
            labelColor="#ffffff",
            tickColor="#ffffff",
        )
        .interactive()             # ← 啟用拖曳、縮放
    )

    st.altair_chart(chart, use_container_width=True)

else:

    st.markdown(
        "<p style='text-align:center; color:#ffffff;background-color: #3b4f32; border-radius:10px; padding:0.6rem 1rem;     '> 請上傳圖片開始診斷</p>",
        unsafe_allow_html=True,
    )


    # 使用說明
    
    with st.expander("使用說明"):
        st.markdown("""
        ### 如何使用本系統

        1. **上傳圖片**：點擊上方的上傳按鈕，選擇植物葉片照片
        2. **等待分析**：系統會自動分析圖片並給出診斷結果
        3. **查看結果**：查看診斷結果、信心度和建議措施
        4. **調整參數**：可在側邊欄調整顯示結果數量和信心度閾值

        ### 拍攝建議

        - 使用清晰的照片
        - 確保光線充足
        - 聚焦在病徵區域
        - 保持適當距離（葉片佔畫面 50-80%）

        ### 支援的病害類別

        本系統可辨識以下 9 種類別：
        - **healthy** (健康)
        - **canker** (潰瘍病)
        - **greasy_spot** (油斑病)
        - **melanose** (黑點病)
        - **sooty_mold** (煤煙病)
        - **pest_thrips** (蟲害－薊馬)
        - **pest_leaf_miner** (蟲害－潛葉蛾)
        - **pest_aphid** (蟲害－蚜蟲 )
        - **pest_scale_insect** (蟲害－介殼蟲 )

        """)

# ========== 頁尾 ==========
acc_text = ""
if model_info.get("accuracy") is not None:
    acc_text = f" | 準確率: {model_info['accuracy']:.2f}%"

st.markdown(f"""
<div style='text-align: center; color: #000000; padding: 1rem;'>
    <p>植物病蟲害智能辨識系統 v1.0</p>
    <p>使用 ConvNeXt Large 深度學習模型{acc_text}</p>
    <p><small>© 2025 - 僅供教學與研究使用</small></p>
</div>
""", unsafe_allow_html=True)