import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정
st.set_page_config(page_title="사문/생윤 문항 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F8F4FF; }
    h1, h2, h3 { color: #6F42C1 !important; }
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; height: 3.5em; width: 100%; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; line-height: 1.8; min-height: 150px; }
    mark { background-color: #E6E0FF; color: #5A32A3; font-weight: bold; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 구글 드라이브 링크 변환 ---
def get_gdrive_direct_link(url):
    file_id = ""
    patterns = [r'/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)', r'srcid=([a-zA-Z0-9_-]+)']
    for p in patterns:
        match = re.search(p, url)
        if match:
            file_id = match.group(1); break
    return f'https://drive.google.com/uc?export=download&id={file_id}' if file_id else url

# --- [사문 특화] 문항 및 선지 통합 추출 로직 ---
def extract_problems_refined(content, filename):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        all_problems = []
        skip_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '문제지', '탐구', '사회·문화', '생활과 윤리']
        
        current_prob_text = ""
        current_num = ""
        current_page = 1

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 'blocks' 모드로 텍스트 덩어리를 순서대로 가져옴
            blocks = page.get_text("blocks", sort=True)
            
            for block in blocks:
                line_text = block[4].replace('\n', ' ').strip()
                if not line_text or len(line_text) < 2: continue
                if any(kw in line_text for kw in skip_keywords): continue

                # [개선된 문항 번호 인식] 
                # 줄의 시작이 숫자 + . 또는 숫자 + ) 또는 [숫자] 인 경우만! 
                # (원문자 ①②③④⑤는 무시하여 제시문-선지가 분리되지 않게 함)
                num_match = re.match(r'^(\d{1,2}[\.|\)]|\[\d{1,2}\])', line_text)
                
                if num_match:
                    # 새로운 문항 번호가 나오면 이전까지 모은 '제시문+선지'를 저장
                    if current_prob_text.strip():
                        all_problems.append({
                            "text": re.sub(r'\s+', ' ', current_prob_text).strip(),
                            "page": current_page,
                            "num": current_num if current_num else "미상",
                            "source": filename
                        })
                    
                    current_num = num_match.group(1).strip()
                    current_prob_text = line_text
                    current_page = page_num + 1
                else:
                    # 문항 번호가 아니면 (선지 번호나 지문 내용이면) 현재 문항에 계속 붙임
                    if current_prob_text:
                        current_prob_text += " " + line_text
                    else:
                        # 번호 없이 시작되는 경우 (예외처리)
                        current_prob_text = line_text
                        current_page = page_num + 1

        # 마지막 문항 저장
        if current_prob_text.strip():
            all_problems.append({
                "text": re.sub(r'\s+', ' ', current_prob_text).strip(),
                "page": current_page,
                "num": current_num if current_num else "마지막",
                "source": filename
            })
            
        return all_problems
    except:
        return []

# --- 하이라이팅 및 분석 로직 ---
def highlight_overlap(target, reference):
    if not target or not reference: return target
    ref_clean = re.sub(r'\s+', '', reference)
    min_len = 6
    to_highlight = []
    for i in range(len(target) - min_len + 1):
        chunk = target[i:i+min_len]
        if len(chunk.strip()) < min_len: continue
        if re.sub(r'\s+', '', chunk) in ref_clean: to_highlight.append(chunk)
    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    result = target
    for chunk in sorted_chunks:
        if chunk in result: result = result.replace(chunk, f"[[MS]]{chunk}[[ME]]")
    return result.replace("[[MS]]", "<mark>").replace("[[ME]]", "</mark>").replace("</mark><mark>", "")

# --- 메인 UI 및 실행 ---
st.title("🟣 사문/생윤 문항 유사도 분석기")

default_links = """모평_수능, https://drive.google.com/file/d/1kf1dZDTFCfAHM9OSAwqaAXI62ClJ3J-S/view?usp=drive_link
2026 수특 생윤, https://drive.google.com/file/d/1xlcMNaNQIbzA1iLXB9lD6eNYL5LM4_LJ/view?usp=drive_link
사문_모평, https://drive.google.com/file/d/1QTIRXZdqlixqhLlUsywqGHZcrxdqZ_mN/view?usp=sharing
2026 사문_수특, https://drive.google.com/file/d/1V-WjvOsOSZwuuRaRObwPqdD07Rvuyx7f/view?usp=drive_link"""

with st.sidebar:
    st.header("🔗 기준 DB 설정")
    links_input = st.text_area("이름, 구글링크", value=default_links, height=200)
    if st.button("🔄 결과 초기화"):
        if 'results' in st.session_state: del st.session_state['results']
        st.rerun()

uploaded_file = st.file_uploader("📝 분석할 대상 PDF 업로드", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 정밀 분석 시작"):
        final_results = []
        all_ref_problems = []
        
        with st.spinner("사문/생윤 문항을 정밀 분석 중입니다..."):
            session = requests.Session()
            lines = [l for l in links_input.split('\n') if ',' in l]
            for line in lines:
                name, url = line.split(',', 1)
                direct_url = get_gdrive_direct_link(url.strip())
                try:
                    res = session.get(direct_url, timeout=60)
                    if res.status_code == 200:
                        all_ref_problems.extend(extract_problems_refined(res.content, name.strip()))
                except: pass

            target_probs = extract_problems_refined(uploaded_file.read(), "업로드")
            
            if all_ref_problems and target_probs:
                prog_bar = st.progress(0)
                vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
                for i, target in enumerate(target_probs):
                    best_score, best_match = 0, None
                    for ref in all_ref_problems:
                        try:
                            tfidf = vectorizer.fit_transform([target['text'], ref['text']])
                            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                            if score > best_score:
                                best_score, best_match = score, ref
                        except: continue
                    final_results.append({
                        "id": i+1, "target": target['text'], "num": target['num'], 
                        "score": round(best_score*100, 1), "match": best_match
                    })
                    prog_bar.progress((i + 1) / len(target_probs))
                
                st.session_state['results'] = final_results
                st.success(f"✅ 분석 완료! {len(target_probs)}개 문항 세트가 인식되었습니다.")

# 결과 표시 영역
if 'results' in st.session_state:
    for res in st.session_state['results']:
        score, match, num = res['score'], res['match'], res['num']
        color = "🔴" if score > 65 else "🟡" if score > 35 else "🟢"
        match_info = f" | [매칭: {match['source']} {match['page']}p {match['num']}]" if match else " | 매칭 없음"
        
        with st.expander(f"{color} {num}번 문항 (유사도 {score}%){match_info}"):
            c1, c2 = st.columns(2)
            if match:
                h_target = highlight_overlap(res['target'], match['text'])
                h_match = highlight_overlap(match['text'], res['target'])
                with c1: st.markdown(f"**[내 문항 (제시문+선지)]**<div class='compare-box'>{h_target}</div>", unsafe_allow_html=True)
                with c2: st.markdown(f"**[DB 문항 (제시문+선지)]**<div class='compare-box'>{h_match}</div>", unsafe_allow_html=True)
            else:
                with c1: st.markdown(f"<div class='compare-box'>{res['target']}</div>", unsafe_allow_html=True)
