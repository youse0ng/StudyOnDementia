import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder, label_binarize
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import plotly.express as px
import time
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve, confusion_matrix


class PlotlyVisualization:
    def __init__(self, data_pkl, selected_data_pkl, modelpkl):
        self.data = joblib.load(data_pkl)
        self.selected_data = joblib.load(selected_data_pkl)
        self.model = joblib.load(modelpkl)
        self.usermodel = None

    def correlation(self, data):
        correlation_matrix = data.corr(numeric_only=True)
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto",
            title=""
        )
        fig.update_layout(
            title="🧠 걸음 라이프로그 특징 간 상관관계 Heatmap",
            title_font=dict(size=18, family='Arial Black', color='black'),
            font=dict(size=13, family='Arial'),
            width=1400,
            height=800,
            margin=dict(l=100, r=50, t=100, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="black",
                font_size=13,
                font_family="Arial",
                font_color="white"
            ),
        )
        fig.update_xaxes(tickangle=75)
        return fig

    def feature_importance(self, top_n=10):
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).head(top_n)

        fig = px.bar(
            df_importance[::-1],
            x='importance',
            y='feature',
            orientation='h',
            title=f'🧠 참조모델이 제공하는 환자를 분류하는 가장 좋은 피쳐 TOP{top_n}',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig.update_layout(
            width=800,
            height=600,
            font=dict(size=14),
            margin=dict(l=150, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(size=18, family='Arial Black', color='black'),
            hoverlabel=dict(
                bgcolor="black",
                font_size=13,
                font_family="Arial",
                font_color="white"
            ),
        )
        return fig

    def Outlier(self):
        def t_scaler(col):
            mean = col.mean()
            var = col.var()
            n = len(col)
            return col.apply(lambda x: (x - mean) / ((var ** 0.5) / (n ** 0.5)))

        copied = self.selected_data.drop(columns='진단명', errors='ignore')
        columns = copied.columns
        t_scaled_data = copied.copy()
        for col in columns:
            t_scaled_data[col] = t_scaler(copied[col])

        fig = go.Figure()
        for col in columns:
            fig.add_trace(go.Box(
                y=t_scaled_data[col],
                name=col,
                boxmean='sd',
                marker=dict(color=np.random.choice(['#FF6347', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2'])),
                line=dict(width=2),
                jitter=0.05
            ))

        fig.update_layout(
            title="🧠T-score 정규화 적용 후 이상치 탐지를 위한 BoxPlot",
            xaxis_title='변수명',
            yaxis_title='T-score',
            template='plotly_white',
            height=600,
            boxmode='group',
            title_font=dict(size=18, family='Arial Black', color='black'),
            hoverlabel=dict(
                bgcolor="black",
                font_size=13,
                font_family="Arial",
                font_color="white"
            ),
        )
        return fig

    def scatter_plot(self, column1='하루간 평균 MET', column2='활동 점수', hue='진단명'):
        if hue not in self.selected_data.columns:
            raise ValueError(f"'{hue}' 컬럼이 존재하지 않습니다.")

        df = self.selected_data[[column1, column2, hue]].copy()
        unique_groups = df[hue].unique()

        # 색상 팔레트
        colors = px.colors.qualitative.Safe  # 잘 구분되는 색상들
        color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}

        fig = go.Figure()

        for group in unique_groups:
            group_df = df[df[hue] == group]
            fig.add_trace(go.Scatter(
                x=group_df[column1],
                y=group_df[column2],
                mode='markers',
                name=str(group),
                marker=dict(
                    size=6,
                    symbol='circle',  # 원형 점
                    color=color_map[group],
                    line=dict(width=0.5, color='black')
                ),
                hovertemplate=(
                    f"<b>{column1}</b>: %{{x:.2f}}<br>"
                    f"<b>{column2}</b>: %{{y:.2f}}<br>"
                    f"<b>{hue}</b>: {group}<extra></extra>"
                )
            ))

        fig.update_layout(
            title = (   f"🧠 2D 산점도<br>"
                        f"🔬X축: {column1}<br>👣Y축: {column2}<br>"
                        f"📌 색상 기준: {hue}"
            ),
            xaxis_title=column1,
            yaxis_title=column2,
            title_font=dict(size=18, family='Arial Black', color='black'),
            template='plotly_dark',  # Cybertic 느낌
            height=600,
            legend_title=hue,
            hoverlabel=dict(
                bgcolor="black",
                font_size=13,
                font_family="Arial",
                font_color="white"
            ),
        )
        return fig

    def scatter_plot_3d(self, column1='하루간 저강도 활동 MET', column2='비활동 시간', column3='고강도 활동 시간', hue='진단명'):
        if hue not in self.data.columns:
            raise ValueError(f"'{hue}' 컬럼이 존재하지 않습니다.")
        df = self.data[[column1, column2, column3, hue]].copy()
        fig = px.scatter_3d(
            df,
            x=column1,
            y=column2,
            z=column3,
            color=hue,
            opacity=0.85,
            color_discrete_sequence=px.colors.qualitative.Safe,  # 시각적으로 뚜렷한 팔레트
            title = (
                        f"🧠 3D 산점도<br>"
                        f"🔬X축: {column1}<br>👣Y축: {column2}<br>🧬Z축: {column3}<br>"
                        f"📌 색상 기준: {hue}"
                    )
        )

        # 모든 마커를 동일한 'circle'로 고정
        fig.update_traces(
            marker=dict(
                size=3.5,                   # 작고 선명하게
                symbol='circle',           # 원형 고정
                line=dict(width=0.5),        # 테두리 제거
                opacity=0.85
            ),
            selector=dict(mode='markers')
        )
        fig.update_layout(
                height=700,
                template='plotly_white',
                legend_title=hue,
                scene=dict(
                    xaxis_title=column1,
                    yaxis_title=column2,
                    zaxis_title=column3
                ),
                    hoverlabel=dict(
                    bgcolor="black",
                    font_size=13,
                    font_family="Arial",
                    font_color="white"
                ),
                title_font=dict(size=18, family='Arial Black', color='black'),
            )
        return fig

    def model_predict(self, selected_model, select_columns):
        X = self.data[select_columns]
        y = self.data['진단명']

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        if selected_model == 'RandomForestClassifier':
            model = RandomForestClassifier(max_depth=6)
        elif selected_model == 'LGBMClassifier':
            model = LGBMClassifier(max_depth=6)
        elif selected_model == 'XGBClassifier':
            model = XGBClassifier()
        else:
            raise ValueError("지원되지 않는 모델입니다.")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return {
            'model': model,
            'feature_importance': getattr(model, 'feature_importances_', None),
            'Accuracy': f"{acc * 100:.2f} %",
            'ClassificationReport': cls_report,
            'ConfusionMatrix': conf_matrix,
            'labels': le.classes_,
            'x_test': X_test,
            'y_test':y_test
        }
    
    def Export_graph(self, selected_model):
        try:
            if not hasattr(selected_model, 'estimators_') or len(selected_model.estimators_) == 0:
                st.warning("트리가 포함된 모델이 아닙니다. RandomForestClassifier 같은 앙상블 모델을 선택하세요.")
                return
            
            max_index = len(selected_model.estimators_) - 1
            tree_index = st.slider("시각화할 트리 인덱스 선택", 0, max_index, 0)

            st.markdown(f"**선택된 트리 번호:** `{tree_index}`")

            feature_names = getattr(selected_model, 'feature_names_in_', None)
            class_names = getattr(selected_model, 'classes_', None)

            if feature_names is None or class_names is None:
                st.warning("모델에 feature_names_in_ 또는 classes_ 속성이 없습니다.")
                return

            dot_data = export_graphviz(
                selected_model.estimators_[tree_index],
                out_file=None,
                feature_names=[str(f) for f in feature_names],
                class_names=[str(c) for c in class_names],
                filled=True,
                rounded=True,
                special_characters=True
            )

            st.graphviz_chart(dot_data)

        except Exception as e:
            st.error(f"그래프 생성 중 오류가 발생했습니다: {e}")

        self.usermodel = selected_model

    def plot_roc_curve(self, model, X_test, y_test):
        # 이진 분류 or One-vs-Rest 방식
        y_score = model.predict_proba(X_test)
        n_classes = y_score.shape[1]

        fig = go.Figure()

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f"ROC Curve (AUC = {roc_auc:.2f})",
                line=dict(color='darkorange', width=2)
            ))

        else:
            # 다중 클래스 → One-vs-Rest 처리
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)

                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"Class {i} (AUC = {roc_auc:.2f})"
                ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title="📉 ROC Curve",
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500,
            legend=dict(x=0.6, y=0.05),
            title_font=dict(size=30),
            hoverlabel=dict(bgcolor="black", font_size=13, font_family="Arial", font_color="white"),
        )

        return fig
    
    def plotly_time_series_with_group_mean(self, data, y='매일 움직인 거리', x='활동 시작 시간', hue='진단명'):
        hue_order = data[hue].unique()
        color_map = px.colors.qualitative.Set1
        color_dict = {label: color_map[i % len(color_map)] for i, label in enumerate(hue_order)}

        # 전체 x축 범위 계산
        x_min = data[x].min()
        x_max = data[x].max()

        fig = go.Figure()
        for group in hue_order:
            group_data = data[data[hue] == group].sort_values(x)
            # 라인 추가
            fig.add_trace(go.Scatter(
                x=group_data[x],
                y=group_data[y],
                mode='lines',
                name=str(group),
                line=dict(color=color_dict[group]),
                marker=dict(size=4),
            ))

            # 전체 x축 기준 평균선 추가
            mean_val = group_data[y].mean()
            fig.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[mean_val, mean_val],
                mode='lines',
                name=f'{group} 평균',
                line=dict(color=color_dict[group], dash='dash'),
                showlegend=True
            ))

        fig.update_layout(
            title=f"📈 {y} (진단명별 시계열 + 평균선)",
            xaxis_title=x,
            yaxis_title=y,
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(title=hue, font=dict(size=12))
        )
        return fig


def main():
    st.set_page_config(
        page_title="치매 환자 데이터 분석 대시보드",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    plotly_vis = PlotlyVisualization('DF.pkl', 'Selected_DF.pkl', 'RF_BESTMODEL.pkl')
    st.title("🧠 치매 환자 분류를 위한 걸음 라이프 로그 데이터 플레이 그라운드")

    def write_stream_caption(text, delay=0.05):
        placeholder = st.empty()
        current_text = ""
        for char in text:
            current_text += char
            # 작고 옅은 텍스트 스타일 (caption 느낌)
            placeholder.markdown(
                f"<p style='font-size: 20px; color: gray;'>{current_text}</p>",
                unsafe_allow_html=True
            )
            time.sleep(delay)

    # 사용 예시
    write_stream_caption("Search Through The Walking LifeLog Data with Your Insight and Hypothesis", delay=0.02)
        
    st.caption("55세 이상 어르신의 걸음 라이프 로그가 담긴 데이터셋입니다.")
    st.subheader("🚶‍♂️ 걸음 라이프 로그 데이터에 대한 설명")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🔹 **하루간 평균 MET**: 하루 전체의 평균적인 에너지 소비량")
        st.markdown("🔹 **하루간 활동 칼로리**: 하루 동안의 활동으로 소비된 총 칼로리")
        st.markdown("🔹 **하루간 총 사용 칼로리**: 전체 에너지 소비량 (기초대사 포함)")
        st.markdown("🔹 **매일 평균적인 거리**: 하루 평균 이동 거리 (단위: km)")
        st.markdown("🔹 **고강도 활동 시간**: 고강도 신체 활동에 사용된 시간 (분)")
        st.markdown("🔹 **비활동 시간**: 거의 움직이지 않은 시간 (분)")
        st.markdown("🔹 **비활동 알림 횟수**: 일정 시간 움직임이 없을 때 받은 알림 횟수")
        st.markdown("🔹 **저강도 활동 시간**: 걷기 등 저강도 활동 시간 (분)")
        st.markdown("🔹 **중강도 활동 시간**: 중간 강도의 활동 시간 (분)")

    with col2:
        st.markdown("🔹 **하루간 고강도 활동 MET**: 고강도 활동의 에너지 소비량")
        st.markdown("🔹 **하루간 비활동 MET**: 비활동 시의 에너지 소비량")
        st.markdown("🔹 **하루간 저강도 활동 MET**: 저강도 활동의 MET 값")
        st.markdown("🔹 **하루간 중강도 활동 MET**: 중강도 활동의 MET 값")
        st.markdown("🔹 **미착용 시간**: 기기를 착용하지 않은 시간")
        st.markdown("🔹 **휴식 시간**: 실제 휴식(수면 외)의 시간")
        st.markdown("🔹 **활동 점수**: 일별 활동량을 점수로 환산한 값")
        st.markdown("🔹 **활동 목표달성 점수**: 설정된 활동 목표에 대한 달성도 점수")
        st.markdown("🔹 **매 시간 당 활동유지 점수**: 일정 수준의 활동을 유지한 정도")

    st.markdown("")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("🔹 **취침시간 점수**: 수면 시작 시점의 일관성과 규칙성 점수")
        st.markdown("🔹 **활동 유지 점수**: 하루 전반에 걸친 일정한 활동 유지 점수")
        st.markdown("🔹 **운동빈도 점수**: 정기적인 운동을 수행한 빈도 점수")

    with col4:
        st.markdown("🔹 **운동강도 점수**: 운동 강도의 평균 수준 점수")
        st.markdown("🔹 **매일 걸음 수**: 하루 평균 걸음 수")
        st.markdown("🔹 **활동 총 시간(분)**: 하루 동안 총 활동에 소비된 시간 (분)")
    st.markdown("---")
    st.subheader("🧠 데이터 시각화")

    # 사이드바 설정
    st.sidebar.header("📊 시각화 설정")
    scatter_col1 = st.sidebar.selectbox("산점도 X축", plotly_vis.data.columns,index=1)
    scatter_col2 = st.sidebar.selectbox("산점도 Y축", plotly_vis.data.columns,index=9)
    scatter_col3 = st.sidebar.selectbox("산점도 Z축 (3D)", plotly_vis.data.columns, index=27)
    top_n_features = st.sidebar.slider("Feature Importance: 상위 N개", min_value=5, max_value=10, value=7)
    model_name = st.sidebar.selectbox(
        "사용할 분류 모델을 선택하세요:",
        ("RandomForestClassifier", "LGBMClassifier", "XGBClassifier")
    )
    # 진단명 컬럼 존재 여부
    hue_col = '진단명' if '진단명' in plotly_vis.selected_data.columns else None

    # 탭 구성
    tabs = st.tabs([
        "📈 상관관계",
        "⭐ 참조모델 Feature 중요도",
        "📦 (T-score) 이상치 박스플롯",
        "⚖️ 산점도 분석",
        "📌 3D 산점도 분석",
        "📉 시계열 데이터 Plot"
    ])

    with tabs[0]:
        st.plotly_chart(plotly_vis.correlation(plotly_vis.data), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plotly_vis.feature_importance(top_n=top_n_features), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(plotly_vis.Outlier(), use_container_width=True)

    with tabs[3]:
        st.plotly_chart(plotly_vis.scatter_plot(column1=scatter_col1, column2=scatter_col2), use_container_width=True)

    with tabs[4]:
        if hue_col:
            st.plotly_chart(plotly_vis.scatter_plot_3d(
                column1=scatter_col1,
                column2=scatter_col2,
                column3=scatter_col3,
                hue=hue_col
            ), use_container_width=True)
        else:
            st.warning("⚠️ '진단명' 컬럼이 없어 그룹별 시각화를 진행할 수 없습니다.")

    with tabs[5]:
        st.plotly_chart(plotly_vis.plotly_time_series_with_group_mean(plotly_vis.data,scatter_col1))

    st.markdown("---")
    st.subheader("🧠 분류 모델 정보 및 성능")

    # 사이드바에서 피처 선택
    columns_to_use = st.sidebar.multiselect(
        "🔍 학습에 사용할 피처(컬럼)를 선택하세요:",
        plotly_vis.data.select_dtypes('number').columns[:-1]
    )

    # 선택한 모델 및 피처 정보 출력
    st.markdown(f"""
    **⚙️ 선택된 분류 모델:** `{model_name}`  
    **🎯 분류 대상 타겟 컬럼:** `{plotly_vis.data.columns[-1]}`  
    **📊 선택된 피처:** `{columns_to_use if columns_to_use else '없음'}`  
    """)

    model = None

    # 피처가 선택되지 않은 경우 경고
    if not columns_to_use:
        st.warning("⚠️ 학습에 사용할 피처를 1개 이상 선택해주세요.")
    else:
        results = plotly_vis.model_predict(model_name, columns_to_use)
        model = results['model']

        # 두 열로 나누어 깔끔하게 표시
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="✅ **Your Accuracy**", value=results['Accuracy'])
    
        with col2:
            try:
                st.write("📌 **Your Feature Importances**", {key:f'{value:.4f}' for key, value in zip(columns_to_use, results['feature_importance'])})
            except:
                st.warning("⚠️까꿍?")
    if model is not None:
        plotly_vis.Export_graph(model)
        # Confusion Matrix
        st.subheader("🧩 Your Confusion Matrix")
        cm_df = pd.DataFrame(results['ConfusionMatrix'], index=results['labels'], columns=results['labels'])

        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
        fig_cm.update_layout(title='Your Confusion Matrix', width=600, height=500)
        st.plotly_chart(fig_cm)

        # Classification Report
        st.subheader("📑 Your Classification Report")
        cls_df = pd.DataFrame(results['ClassificationReport']).transpose().round(2)
        st.dataframe(cls_df.style.background_gradient(cmap='YlGnBu'), height=300)

        fig_roc = plotly_vis.plot_roc_curve(plotly_vis.usermodel, results['x_test'], results['y_test'])
        st.plotly_chart(fig_roc)
    
if __name__ == "__main__":
    main()
