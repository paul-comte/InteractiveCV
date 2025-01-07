from pathlib import Path
import streamlit as st
from PIL import Image

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "Presentation.css"
resume_file = current_dir / "assets" / "CV - Paul Comte - 05-25.pdf"
profile_picture = current_dir / "assets" / "cv_picture.jpg"


#GENERAL settings

PAGE_TITLE = "Digital CV - Paul Comte"
PAGE_ICON = ":wave:"
NAME = "Paul Comte"
DESCRIPTION = """
EDHEC Student - MSc in Financial Markets \n
I am looking for a 6-month internship starting in June - September 2025 in Quantitative Finance
"""

EMAIL = "paul.comte@edhec.com"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/paulcomte/",
    "GitHub": "https://github.com/paul-comte"
}

st.set_page_config(page_title = PAGE_TITLE, page_icon = PAGE_ICON)

#-- LOAD CSS, PDF & PROfILE PICTURE

with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_picture = Image.open(profile_picture)

#-- HERO SECTION --
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_picture, width=230)
with col2:
    st.title(NAME)
    st.markdown(DESCRIPTION)
    st.download_button(
        label="Download Resume 📝",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/pdf"
    )
    st.write(f"Email 📧 : {EMAIL}")

#-- SOCIAL LINKS --
st.write('#')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    with cols[index]:
        st.markdown(f"[{platform}]({link})")
        
#-- SKILLS -- 
st.write('#')
st.subheader("CODING SKILLS 👨🏻‍💻")
st.write("""
**Coding** - Python, VBA, SQL\n
- Python projects on machine learning, data analysis, and financial modeling : [GitHub](https://github.com/paul-comte)\n
- VBA : 20/20 at the EDHEC exam\n      
- Actually learning SQL\n
""")
st.subheader("FINANCIAL SKILLS 📈")
st.write("""
\n\n**Financial Markets** - Derivatives, Fixed Income, Portfolio Management\n
""")

#-- PRESENTATION --
st.write('#')
st.subheader("Presentation 👨🏻‍🏫")
st.write("""
I am a student at EDHEC Business School in the MSc in Financial Markets. I am looking for a 6-month internship starting in June - September 2025 in Quantitative Finance. I am passionate about financial markets and I am looking for a position that will allow me to deepen my knowledge in this field. I am also interested in the use of data science in finance and I am currently learning Python and SQL. I am a hardworking and motivated person who is always looking for new challenges. I am also a team player and I am used to working in a multicultural environment. On this website, you will find my projects on Python. Feel free to try them and give me your feedback. Do not hesitate to contact me if you have any questions or if you want to know more about me.
""")