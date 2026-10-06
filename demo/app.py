"""
Streamlit 可视化 Demo
"""
import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="小红书推荐 Demo", layout="wide")
st.title("📕 小红书风格笔记推荐系统")
st.caption("多路召回 + LightGBM 排序 + MMR 多样性重排")

# 侧边栏
with st.sidebar:
    st.header("参数")
    user_id = st.number_input("用户 ID", min_value=0, max_value=999, value=0)
    k = st.slider("推荐数量", 5, 20, 10)

    if st.button("🔄 刷新缓存"):
        st.session_state.pop("result", None)

# 主区域
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("服务状态")
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success("✅ 服务正常")
            st.metric("笔记数", data["n_notes"])
            st.metric("用户数", data["n_users"])
        else:
            st.error(f"服务异常: {r.status_code}")
    except Exception as e:
        st.error(f"无法连接服务: {e}")
        st.code("uvicorn service.main:app --port 8000", language="bash")

with col2:
    st.subheader(f"用户 {user_id} 的推荐结果")

    if st.button("🚀 获取推荐", type="primary"):
        with st.spinner("调用推荐服务..."):
            try:
                resp = requests.post(
                    f"{API}/recommend",
                    json={"user_id": int(user_id), "k": int(k)},
                    timeout=10,
                )
                if resp.status_code == 200:
                    st.session_state["result"] = resp.json()
                else:
                    st.error(f"错误 {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"请求失败: {e}")

    if "result" in st.session_state:
        data = st.session_state["result"]
        st.caption(
            f"延迟: {data['latency_ms']:.1f} ms | "
            f"缓存: {'是' if data['from_cache'] else '否'}"
        )

        for i, item in enumerate(data["items"], 1):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 6, 2])
                with c1:
                    st.markdown(f"### #{i}")
                with c2:
                    st.markdown(f"**{item['title']}**")
                    st.caption(
                        f"🆔 {item['note_id']}  |  "
                        f"📁 {item['category']}  |  "
                        f"🏷️ {item['tags']}"
                    )
                with c3:
                    st.metric("score", f"{item['score']:.4f}")
                    st.caption(f"来源: {item['source']}")