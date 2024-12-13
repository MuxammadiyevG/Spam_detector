import streamlit as st
import pickle

# Sahifani sozlash
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Sahifa sarlavhasi va izoh
st.title("📧 Spam Detector")
st.markdown("""
<div style="background-color:#f7f7f9; padding:15px; border-radius:5px; margin-bottom:15px;">
    <p style="font-size:16px; font-family:sans-serif; color:#4a4a4a;">
        Ushbu model sizga Email xabarini spam yoki spam emasligini aniqlashga yordam beradi.
    </p>
</div>
""", unsafe_allow_html=True)

# Model va vektorizatorni yuklash
model_file_path = "spam_model.pkl"
vector_file = "vectorizer.pkl"

try:
    with open(model_file_path, 'rb') as fs:
        model = pickle.load(fs)
    with open(vector_file, 'rb') as vs:
        vectorizer = pickle.load(vs)
except Exception as e:
    st.error(f"Modelni yuklashda xato: {e}")

# Foydalanuvchi xabarini kiritish
email = st.text_area("Xabarni kiriting", placeholder="Bu yerga email xabarini kiriting...", height=150)

# "Yuborish" tugmasi
if st.button("Yuborish ✉️"):
    if email.strip():  # Xabar bo'sh emasligini tekshirish
        vect = vectorizer.transform([email])
        pred = model.predict(vect)

        # Natijani ko'rsatish
        if pred[0] == 1:
            st.markdown("""
            <div style="background-color:#ffdddd; padding:15px; border-radius:5px; margin-top:15px;">
                <p style="font-size:18px; font-family:sans-serif; color:#d9534f;">
                    Bu xabar <b>spam</b> xabar!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:#ddffdd; padding:15px; border-radius:5px; margin-top:15px;">
                <p style="font-size:18px; font-family:sans-serif; color:#5cb85c;">
                    Bu xabar <b>spam emas</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Iltimos, xabar kiriting!")
