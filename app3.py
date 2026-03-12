import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd
from newspaper import Article, Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [1단계] 문장 분리 및 형광펜 하이라이트 알고리즘 ---
def split_sentences(text):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def get_highlighted_texts(text1, text2, threshold=0.4):
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)

    if not sents1 or not sents2:
        return text1, text2

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sents1 + sents2)
        mat1 = tfidf_matrix[:len(sents1)]
        mat2 = tfidf_matrix[len(sents1):]
        sim_matrix = cosine_similarity(mat1, mat2)
    except ValueError:
        return text1, text2

    highlighted_sents1 = []
    for i, s in enumerate(sents1):
        if len(sim_matrix[i]) > 0 and max(sim_matrix[i]) >= threshold:
            highlighted_sents1.append(f'<mark style="background-color: #fff3cd; color: #856404; font-weight: bold; border-radius: 3px; padding: 2px;">{s}</mark>')
        else:
            highlighted_sents1.append(s)

    highlighted_sents2 = []
    for j, s in enumerate(sents2):
        if len(sents1) > 0 and max([sim_matrix[i][j] for i in range(len(sents1))]) >= threshold:
            highlighted_sents2.append(f'<mark style="background-color: #fff3cd; color: #856404; font-weight: bold; border-radius: 3px; padding: 2px;">{s}</mark>')
        else:
            highlighted_sents2.append(s)

    return " ".join(highlighted_sents1), " ".join(highlighted_sents2)

# --- [2단계] 구글 뉴스 크롤링 ---
def crawl_related_articles(query, num_results=10):
    crawled_articles = []
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
        
        user_config = Config()
        user_config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        user_config.request_timeout = 5 
        
        for item in items[:num_results]:
            title = item.find('title').text
            url = item.find('link').text
            
            try:
                article = Article(url, config=user_config, language='ko')
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 50:
                    crawled_articles.append({
                        'title': title,
                        'url': url,
                        'text': article.text
                    })
            except Exception:
                continue
    except Exception as e:
        st.error(f"🚨 사내 방화벽으로 인해 구글 뉴스 접근이 차단되었거나 지연되었습니다.")
    return crawled_articles

# --- [3단계] 유사도 계산 ---
def calculate_similarity(my_title, my_text, crawled_data):
    my_full_article = my_title + "\n" + my_text
    texts = [my_full_article] + [item['title'] + "\n" + item['text'] for item in crawled_data]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    results = []
    for idx, item in enumerate(crawled_data):
        results.append({
            '기사 제목': item['title'],
            '유사도 (%)': round(cosine_sim[idx] * 100, 1),
            '기사 링크': item['url'],
            '타사 본문': item['text'] 
        })
    return results

# --- 메인 웹 앱 로직 ---
def main():
    st.set_page_config(page_title="경인방송 기사 분석기 (최종)", page_icon="📰", layout="wide")
    
    st.title("📰 경인방송 보도국 - 기사 유사도 검증 시스템")
    st.markdown("---")
    
    st.subheader("1️⃣ 내 기사 입력")
    article_title = st.text_input("📌 기사 제목:", placeholder="예: [단독] 경인방송, 새로운 AI 시스템 도입...")
    article_text = st.text_area("📝 기사 본문:", height=200, placeholder="검사할 기사 내용을 여기에 붙여넣으세요...")

    st.markdown("---")
    st.subheader("2️⃣ 비교 방식 선택")
    compare_mode = st.radio(
        "어떤 방식으로 타사 기사와 비교하시겠습니까?",
        ("🌐 구글 뉴스 자동 검색하여 비교하기", "📝 특정 타사 기사를 직접 입력하여 비교하기")
    )
    st.markdown("<br>", unsafe_allow_html=True)

    my_full_text = article_title + "\n" + article_text

    # ==========================================
    # 모드 A: 구글 뉴스 검색
    # ==========================================
    if compare_mode == "🌐 구글 뉴스 자동 검색하여 비교하기":
        st.markdown("🔑 **비교 검색에 사용할 핵심 키워드 (최대 5개)**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: kw1 = st.text_input("키워드 1 (필수)")
        with col2: kw2 = st.text_input("키워드 2 (선택)")
        with col3: kw3 = st.text_input("키워드 3 (선택)")
        with col4: kw4 = st.text_input("키워드 4 (선택)")
        with col5: kw5 = st.text_input("키워드 5 (선택)")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 자동 검색 및 비교 분석 시작", use_container_width=True):
            if not article_text.strip():
                st.warning("⚠️ 내 기사 본문을 먼저 입력해 주세요!")
                return
            entered_keywords = [kw.strip() for kw in [kw1, kw2, kw3, kw4, kw5] if kw.strip()]
            if not entered_keywords:
                st.warning("⚠️ 검색을 위해 최소 1개 이상의 키워드를 입력해 주세요!")
                return

            search_query = " ".join(entered_keywords)
            
            with st.spinner("구글 뉴스에서 타사 기사를 수집 중입니다... (최대 5~10초 대기)"):
                crawled_data = crawl_related_articles(search_query, num_results=10)
            
            if not crawled_data:
                st.warning("비교할 타사 기사를 찾지 못했습니다.")
                return

            with st.spinner(f"총 {len(crawled_data)}개의 기사를 찾았습니다! 유사도 분석 중..."):
                results = calculate_similarity(article_title, article_text, crawled_data)
                
                # 🚨 [수정됨] 분석 결과를 메모리(session_state)에 저장!
                st.session_state['df_results_A'] = pd.DataFrame(results).sort_values(by='유사도 (%)', ascending=False)
                st.session_state['my_full_text_A'] = my_full_text

        # 🚨 [수정됨] 메모리에 결과가 저장되어 있다면 언제든 화면을 다시 그립니다.
        if 'df_results_A' in st.session_state:
            df_results = st.session_state['df_results_A']
            saved_my_text = st.session_state['my_full_text_A']
            display_df = df_results[['기사 제목', '유사도 (%)', '기사 링크']]

            st.markdown("### 📊 분석 결과")
            tab1, tab2, tab3 = st.tabs(["📋 세부 결과 표", "📈 유사도 그래프", "🖍️ 문장별 상세 비교 (형광펜)"])
            
            with tab1:
                st.dataframe(display_df, column_config={"기사 링크": st.column_config.LinkColumn("원문 링크")}, use_container_width=True)
            with tab2:
                chart_data = display_df.set_index('기사 제목')['유사도 (%)']
                st.bar_chart(chart_data)
            
            with tab3:
                st.info("💡 클릭하여 아래로 펼치면, 두 기사에서 의미가 비슷하거나 일치하는 문장이 노란색으로 표시됩니다.")
                
                # 이제 슬라이더를 마음껏 움직여도 화면이 사라지지 않습니다!
                highlight_threshold = st.slider(
                    "🎚️ 형광펜 민감도 조절 (숫자가 낮을수록 예민하게 반응합니다)", 
                    min_value=10, max_value=90, value=20, step=5, key="slider_A"
                ) / 100.0 
                
                for index, row in df_results.iterrows():
                    with st.expander(f"🔽 {row['기사 제목']} (전체 유사도: {row['유사도 (%)']}%)"):
                        high_my_text, high_other_text = get_highlighted_texts(
                            saved_my_text, row['타사 본문'], threshold=highlight_threshold
                        )
                        
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown("#### 📝 내 기사")
                            st.markdown(high_my_text, unsafe_allow_html=True)
                        with col_right:
                            st.markdown("#### 🎯 타사 기사")
                            st.markdown(high_other_text, unsafe_allow_html=True)

    # ==========================================
    # 모드 B: 직접 입력
    # ==========================================
    elif compare_mode == "📝 특정 타사 기사를 직접 입력하여 비교하기":
        st.markdown("**비교할 타 언론사의 기사를 아래에 붙여넣어 주세요.**")
        other_article_text = st.text_area("🎯 타사 기사 본문:", height=200, placeholder="비교할 타사 기사 내용을 여기에 붙여넣으세요...")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 1:1 직접 비교 분석 시작", use_container_width=True):
            if not article_text.strip() or not other_article_text.strip():
                st.warning("⚠️ 내 기사와 타사 기사 본문을 모두 입력해 주세요!")
                return
            
            with st.spinner("두 기사의 유사도를 정밀하게 비교하고 있습니다..."):
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform([my_full_text, other_article_text])
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()
                    sim_score = round(cosine_sim[0] * 100, 1)
                except ValueError:
                    sim_score = 0.0

                # 🚨 [수정됨] 1:1 결과도 메모리에 단단히 저장!
                st.session_state['sim_score_B'] = sim_score
                st.session_state['my_full_text_B'] = my_full_text
                st.session_state['other_article_text_B'] = other_article_text

        # 🚨 [수정됨] 메모리에서 불러와서 화면 그리기
        if 'sim_score_B' in st.session_state:
            saved_sim_score = st.session_state['sim_score_B']
            saved_my_text = st.session_state['my_full_text_B']
            saved_other_text = st.session_state['other_article_text_B']

            st.markdown("### 📊 1:1 분석 결과")
            st.metric(label="두 기사의 텍스트 유사도", value=f"{saved_sim_score}%")
            st.progress(int(saved_sim_score))
            
            st.markdown("---")
            st.markdown("### 🖍️ 문장별 상세 비교 (형광펜)")
            st.info("💡 두 기사에서 의미가 비슷하거나 일치하는 문장이 노란색으로 칠해집니다.")
            
            # 슬라이더 조작에도 끄떡없습니다.
            highlight_threshold = st.slider(
                "🎚️ 형광펜 민감도 조절 (숫자가 낮을수록 예민하게 반응합니다)", 
                min_value=10, max_value=90, value=20, step=5, key="slider_B"
            ) / 100.0 
            
            high_my_text, high_other_text = get_highlighted_texts(
                saved_my_text, saved_other_text, threshold=highlight_threshold
            )
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### 📝 내 기사")
                st.markdown(high_my_text, unsafe_allow_html=True)
            with col_right:
                st.markdown("#### 🎯 타사 기사")
                st.markdown(high_other_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
