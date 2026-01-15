import streamlit as st
import fitz
import re
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F8F4FF; }
    h1, h2, h3 { color: #6F42C1 !important; }
    div.stButton > button { background-color: #6F42C1; color: white; border-radius: 10px; font-weight: bold; height: 3.5em; }
    .compare-box { border: 2px solid #E0D4F7; padding: 20px; border-radius: 15px; background-color: white; line-height: 1.8; }
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

# --- [정밀도 강화] 텍스트 추출 (이미지/레이아웃 대응) ---
def extract_problems_refined(content, filename):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        all_problems = []
        skip_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '문제지', '탐구', '사회·문화']
        
        current_prob = ""
        current_num = ""
        current_page = 1

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # sort=True를 사용하여 이미지가 큰 페이지에서도 읽기 순서대로 텍스트 정렬
            text_blocks = page.get_text("blocks", sort=True)
            
            for block in text_blocks:
                line_text = block[4].replace('\n', ' ').strip() # 블록 내 줄바꿈 제거
                if not line_text or len(line_text) < 2: continue
                if any(kw in line_text for kw in skip_keywords): continue

                # 문항 번호 감지 강화 (예: 1. [1] 1) ① 등과 겹치지 않게)
                num_match = re.match(r'^(\d+[\.|\)]|\[\d+\])', line_text)
                
                if num_match:
                    if current_prob.strip():
                        all_problems.append({
                            "text": current_prob.strip(),
                            "page": current_page,
                            "num": current_num if current_num else "미상",
                            "source": filename
                        })
                    current_num = num_match.group(1).strip()
                    current_prob = line_text
                    current_page = page_num + 1
                else:
                    if current_prob:
                        current_prob += " " + line_text
                    else:
                        current_prob = line_text
                        current_page = page_num + 1

        # 마지막 문항 저장
        if current_prob.strip():
            all_problems.append({
                "text": current_prob.strip(),
                "page": current_page,
                "num": current_num if current_num else "마지막",
                "source": filename
            })
            
        return all_problems
    except Exception as e:
        return []

# --- 하이라이팅 로직 ---
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

# --- 메인 실행부 ---
st.title("🟣 문항 유사도 분석기")

# [수정] 사회문화(사문) 링크 2개 고정값 추가
default_links = """모평_수능, https://drive.google.com/file/d/1kf1dZDTFCfAHM9OSAwqaAXI62ClJ3J-S/view?usp=drive_link
2026 수특 생윤, https://drive.google.com/file/d/1xlcMNaNQIbzA1iLXB9lD6eNYL5LM4_LJ/view?usp=drive_link
사문_모평, https://drive.google.com/file/d/1QTIRXZdqlixqhLlUsywqGHZcrxdqZ_mN/view?usp=sharing
2026 사문_수특, https://drive.google.com/file/d/1V-WjvOsOSZwuuRaRObwPqdD07Rvuyx7f/view?usp=drive_link"""

with st.sidebar:
    st.header("🔗 기준 DB 등록")
    links_input = st.text_area("이름, 구글링크", value=default_links, height=200)

uploaded_file = st.file_uploader("📝 분석할 문항 PDF 업로드", type="pdf")

if uploaded_file and links_input:
    if st.button("🚀 정밀 분석 시작"):
        final_results = []
        all_ref_problems = []
        status_msg = st.empty()
        
        session = requests.Session()
        lines = [line for line in links_input.split('\n') if ',' in line]
        
        for line in lines:
            name, url = line.split(',', 1)
            name = name.strip()
            status_msg.info(f"⏳ '{name}' 데이터를 가져오는 중...")
            direct_url = get_gdrive_direct_link(url.strip())
            try:
                res = session.get(direct_url, timeout=60) # 이미지 대비 타임아웃 60초 연장
                if res.status_code == 200:
                    all_ref_problems.extend(extract_problems_refined(res.content, name))
            except: pass

        if all_ref_problems:
            target_probs = extract_problems_refined(uploaded_file.read(), "업로드")
            
            if target_probs:
                prog = st.progress(0)
                vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
                
                for i, target in enumerate(target_probs):
                    t_num = target.get('num', '미상')
                    status_msg.info(f"🔍 {i+1}번({t_num}) 문항 대조 중...")
                    
                    best_score, best_match = 0, None
                    for ref in all_ref_problems:
                        try:
                            tfidf = vectorizer.fit_transform([target['text'], ref['text']])
                            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                            if score > best_score:
                                best_score, best_match = score, ref
                        except: continue
                    
                    final_results.append({
                        "id": i + 1, "target": target['text'], "num": t_num,
                        "score": round(best_score*100, 1), "match": best_match
                    })
                    prog.progress((i + 1) / len(target_probs))
                
                st.session_state['results'] = final_results
                status_msg.success(f"✅ 총 {len(target_probs)}개 문항 분석 완료!")

# 결과 표시
if 'results' in st.session_state:
    st.markdown("---")
    for res in st.session_state['results']:
        score, match, num = res['score'], res['match'], res['num']
        color = "🔴" if score > 65 else "🟡" if score > 35 else "🟢"
        info = f" - [매칭: {match['source']} {match['page']}p {match['num']}]" if match else ""
        
        with st.expander(f"{color} {num}번 (유사도 {score}%){info}"):
            c1, c2 = st.columns(2)
            h_target = highlight_overlap(res['target'], match['text']) if match else res['target']
            with c1: st.markdown(f"<div class='compare-box'><b>[대상 문항]</b><hr>{h_target}</div>", unsafe_allow_html=True)
            if match:
                h_match = highlight_overlap(match['text'], res['target'])
                with c2: st.markdown(f"<div class='compare-box'><b>[DB 문항]</b><hr>{h_match}</div>", unsafe_allow_html=True)
