import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd
from newspaper import Article, Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [1단계] 구글 뉴스 RSS 기반 크롤링 ---
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

# --- [2단계-A] 유사도 계산 (여러 기사 비교용) ---
def calculate_similarity(my_title, my_text, crawled_data):
    my_full_article = my_title + " " + my_text
    texts = [my_full_article] + [item['title'] + " " + item['text'] for item in crawled_data]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    results = []
    for idx, item in enumerate(crawled_data):
        results.append({
            '기사 제목': item['title'],
            '유사도 (%)': round(cosine_sim[idx] * 100, 1),
            '기사 링크': item['url']
        })
    return results

# --- [2단계-B] 1:1 직접 유사도 계산 (단일 기사 비교용) ---
def calculate_direct_similarity(text1, text2):
    """두 텍스트를 직접 1:1로 비교하여 퍼센트를 반환합니다."""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()
        return round(cosine_sim[0] * 100, 1)
    except ValueError:
        return 0.0

# --- 메인 웹 앱 로직 ---
def main():
    st.set_page_config(page_title="경인방송 기사 분석기 (Dual 모드)", page_icon="📰", layout="wide")
    
    st.title("📰 경인방송 보도국 - 기사 유사도 검증 시스템")
    st.markdown("---")
    
    # 1. 내 기사 입력 영역 (항상 보임)
    st.subheader("1️⃣ 내 기사 입력")
    article_title = st.text_input("📌 기사 제목:", placeholder="예: [단독] 경인방송, 새로운 AI 시스템 도입...")
    article_text = st.text_area("📝 기사 본문:", height=200, placeholder="검사할 기사 내용을 여기에 붙여넣으세요...")

    st.markdown("---")
    
    # 2. 비교 모드 선택 (라디오 버튼)
    st.subheader("2️⃣ 비교 방식 선택")
    compare_mode = st.radio(
        "어떤 방식으로 타사 기사와 비교하시겠습니까?",
        ("🌐 구글 뉴스 자동 검색하여 비교하기", "📝 특정 타사 기사를 직접 입력하여 비교하기")
    )

    st.markdown("<br>", unsafe_allow_html=True)

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
                st.warning("비교할 타사 기사를 찾지 못했습니다. 언론사 보안에 막혔거나 검색 결과가 없습니다.")
                return

            with st.spinner(f"총 {len(crawled_data)}개의 기사를 찾았습니다! 유사도 분석 중..."):
                results = calculate_similarity(article_title, article_text, crawled_data)
                df_results = pd.DataFrame(results).sort_values(by='유사도 (%)', ascending=False)

            st.markdown("### 📊 분석 결과")
            tab1, tab2 = st.tabs(["📋 세부 결과 표", "📈 유사도 그래프"])
            with tab1:
                st.dataframe(df_results, column_config={"기사 링크": st.column_config.LinkColumn("원문 링크")}, use_container_width=True)
            with tab2:
                chart_data = df_results.set_index('기사 제목')['유사도 (%)']
                st.bar_chart(chart_data)

    # ==========================================
    # 모드 B: 직접 입력
    # ==========================================
    elif compare_mode == "📝 특정 타사 기사를 직접 입력하여 비교하기":
        st.markdown("**비교할 타 언론사의 기사를 아래에 붙여넣어 주세요.** (사내 방화벽의 영향을 전혀 받지 않습니다.)")
        other_article_text = st.text_area("🎯 타사 기사 본문:", height=200, placeholder="비교할 타사 기사 내용을 여기에 붙여넣으세요...")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 1:1 직접 비교 분석 시작", use_container_width=True):
            if not article_text.strip():
                st.warning("⚠️ 내 기사 본문을 먼저 입력해 주세요!")
                return
            if not other_article_text.strip():
                st.warning("⚠️ 비교할 타사 기사 본문을 입력해 주세요!")
                return
            
            with st.spinner("두 기사의 유사도를 정밀하게 비교하고 있습니다..."):
                # 1:1 유사도 계산
                sim_score = calculate_direct_similarity(article_title + " " + article_text, other_article_text)
            
            st.markdown("### 📊 1:1 분석 결과")
            
            # 보기 좋게 큰 글씨와 게이지(Progress bar)로 표현
            st.metric(label="두 기사의 텍스트 유사도", value=f"{sim_score}%")
            st.progress(int(sim_score))
            
            if sim_score >= 50:
                st.error("🚨 유사도가 상당히 높습니다. 문장 구조나 단어 선택을 수정하는 것을 권장합니다.")
            elif sim_score >= 30:
                st.warning("⚠️ 일부 문장이나 핵심 단어가 겹칠 수 있습니다. 참고 바랍니다.")
            else:
                st.success("✅ 유사도가 낮습니다. 독창적인 기사로 보입니다!")

if __name__ == "__main__":
    main()