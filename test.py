import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pandas as pd
import plotly.express as px

class PlotlyVisualization:
    def __init__(self,data_pkl, selected_data_pkl, modelpkl):
        self.data = joblib.load(data_pkl)
        self.selected_data = joblib.load(selected_data_pkl)
        self.model = joblib.load(modelpkl)

    def correlation(self, data):
        correlation_matrix = data.corr(numeric_only=True)

        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto",
            title=" "
        )
        fig.update_layout(
            title_font=dict(size=24, family='Arial Black', color='black'),
            font=dict(size=13, family='Arial'),
            xaxis=dict(
                tickangle=75,
                tickfont=dict(size=12),
                side='top'
            ),
            yaxis=dict(tickfont=dict(size=12)),
            width=1400,  # 가로 크기 확대
            height=800,
            margin=dict(l=100, r=50, t=100, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            coloraxis_colorbar=dict(
                title="상관계수",
                ticks="outside",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                lenmode="pixels",
                len=300,
                thickness=15,
                title_side='right'
            )
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig
    
    def feature_importance(self, top_n=10):
        # 모델에 feature_importances_ 속성 있는지 확인
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")

        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        # 중요도 DataFrame 생성
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # 중요도 내림차순 정렬 및 상위 n개 선택
        df_importance = df_importance.sort_values(by='importance', ascending=False).head(top_n)

        # plotly 막대그래프 (가로)
        fig = px.bar(
            df_importance[::-1],  # 역순으로 그래프 그려서 중요도 큰 게 위로
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )

        fig.update_layout(
            width=800,
            height=600,
            font=dict(size=14),
            margin=dict(l=150, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def Outlier(self):
        # Z-score 정규화
        scaler = StandardScaler()
        self.selected_data = self.selected_data.drop(columns='진단명')
        columns = self.selected_data.columns
        self.selected_data = pd.DataFrame(scaler.fit_transform(self.selected_data),columns=columns)
        # Plotly Boxplot
        fig = go.Figure()

        # 각 변수에 대해 Boxplot을 추가
        for col in columns:
            fig.add_trace(go.Box(
                y=self.selected_data[col],
                name=col,
                boxmean='sd',  # 평균과 표준편차 추가
                marker=dict(color=np.random.choice(['#FF6347', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2'])),  # 색상 지정
                line=dict(width=2),
                jitter=0.05
            ))

        # 레이아웃 설정
        fig.update_layout(
            title="Numerical Columns - Boxplot (Z-score 정규화)",
            title_font=dict(size=24),
            xaxis_title='변수명',
            yaxis_title='정규화된 값',
            template='plotly_white',
            boxmode='group',  # Boxplot을 그룹화
            height=600,
            showlegend=True
        )
        return fig

    def scatter_plot(self,column1='하루간 평균 MET',column2='활동 점수'):
        x = self.selected_data[column1]
        y = self.selected_data[column2]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',  # 점 + 라벨 (옵션)
            marker=dict(size=10, color='royalblue', line=dict(width=2, color='DarkSlateGrey')),
            textposition='top center',
            name='관측치'
        ))

        fig.update_layout(
            title='하루간 평균 MET vs 활동 점수',
            xaxis_title='하루간 평균 MET',
            yaxis_title='활동 점수',
            template='plotly_white',
            height=500
        )
        return fig
    
def main():
    plotly_vis = PlotlyVisualization('DF.pkl','Selected_DF.pkl','RF_BESTMODEL.pkl')
    st.title("치매 환자 분류를 위한 데이터 분석")
    st.plotly_chart(plotly_vis.correlation(plotly_vis.data),use_container_width=False)
    st.plotly_chart(plotly_vis.feature_importance(),use_container_width=False)
    st.plotly_chart(plotly_vis.Outlier(),use_container_width=False)
    st.plotly_chart(plotly_vis.scatter_plot(),use_container_width=False)
if __name__ == "__main__":
    main()
