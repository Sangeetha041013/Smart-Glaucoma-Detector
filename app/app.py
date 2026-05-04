# app/app.py
import os
import sys
import streamlit as st
from PIL import Image, ImageFilter, ImageStat
import hashlib
import json
import time
import numpy as np
import cv2

# so we can import from src/
sys.path.append("src")
from predict_single import load_model, predict_image

RESNET_MODEL_PATH = "models/resnet50_glaucoma.pth"
DENSENET_MODEL_PATH = "models/densenet121_glaucoma.pth"

# ----------------------------------------------------------
# IMAGE VALIDATION FUNCTIONS
# ----------------------------------------------------------
def is_fundus_image(image):
    """
    Validate if the uploaded image is a retinal fundus image.
    Returns: (is_valid, reason_message)
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check image dimensions
        height, width = img_array.shape[:2]
        
        # Fundus images are typically circular/elliptical and have certain size ranges
        if height < 200 or width < 200:
            return False, "Image too small. Fundus images are typically larger."
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Check for circular/elliptical patterns (fundus images usually have dark borders)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Fundus images typically have circular edges
        if edge_pixels < 100:
            return False, "Image lacks clear retinal structure."
        
        # Check color distribution - fundus images have specific color ranges
        if len(img_array.shape) == 3:
            # Check red channel intensity (fundus images are reddish)
            red_channel = img_array[:, :, 0]
            green_channel = img_array[:, :, 1]
            
            red_mean = np.mean(red_channel)
            green_mean = np.mean(green_channel)
            
            # Fundus images usually have more red than green
            if red_mean < green_mean * 0.8:
                return False, "Color pattern doesn't match retinal fundus images."
        
        # Check for brightness - fundus images have specific lighting
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 230:
            return False, "Wrong image uploaded. Please upload a valid retinal fundus image."
        
        # Check for typical fundus features using edge density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Fundus images have moderate texture
        texture_score = np.mean(gradient_magnitude)
        if texture_score < 10:
            return False, "Image texture doesn't match retinal patterns."
        
        # All checks passed
        return True, "Image appears to be a retinal fundus image."
        
    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"

def validate_fundus_image(image):
    """
    Perform multiple validations on the image
    """
    # Check 1: Basic image properties
    if image.mode not in ['RGB', 'L', 'RGBA']:
        return False, "Unsupported image format. Please use RGB images."
    
    # Check 2: Size validation
    if image.size[0] < 100 or image.size[1] < 100:
        return False, "Image too small. Minimum size: 100x100 pixels."
    
    # Check 3: Aspect ratio (fundus images are typically square-ish)
    aspect_ratio = image.size[0] / image.size[1]
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False, "Wrong image uploaded. Please upload a valid retinal fundus image."
    
    # Check 4: Use advanced validation
    return is_fundus_image(image)

# ----------------------------------------------------------
# USER AUTHENTICATION FUNCTIONS
# ----------------------------------------------------------
USER_DATA_FILE = "users.json"

def load_users():
    """Load user data from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save user data to JSON file"""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email="", role="user"):
    """Register a new user"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        "password_hash": hash_password(password),
        "email": email,
        "role": role,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "login_count": 0
    }
    
    save_users(users)
    return True, "Registration successful"

def authenticate_user(username, password):
    """Authenticate user login"""
    users = load_users()
    
    if username not in users:
        return False, "Invalid username or password"
    
    user = users[username]
    
    if user["password_hash"] != hash_password(password):
        return False, "Invalid username or password"
    
    # Update login count
    user["login_count"] = user.get("login_count", 0) + 1
    user["last_login"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_users(users)
    
    return True, "Login successful"

# ----------------------------------------------------------
# DOCTORS DATA
# ----------------------------------------------------------
KARNATAKA_DOCTORS = [
    {
        "name": "Dr. Rohit Shetty",
        "specialization": "Glaucoma Specialist & Cataract Surgeon",
        "hospital": "Narayana Nethralaya",
        "city": "Bengaluru",
        "address": "Narayana Health City, #258/A, Bommasandra Industrial Area, Hosur Road",
        "phone": "+91-80-6612 1300",
        "experience": "25+ years",
        "qualifications": "MBBS, MS, FRCS (Glasgow), DNB",
        "rating": "4.8/5",
        "appointment": "www.narayananethralaya.com"
    },

    {
        "name": "Dr. K. Bhujang Shetty",
        "specialization": "Glaucoma & Cataract Surgery",
        "hospital": "Shetty Eye Centre",
        "city": "Mangaluru",
        "address": "Kodialbail, Opposite City Hospital",
        "phone": "+91-824-244 0744",
        "experience": "28+ years",
        "qualifications": "MBBS, MS (Ophthalmology)",
        "rating": "4.6/5",
        "appointment": "shettyeye.com"
    },
    
    {
        "name": "Dr. S. Natarajan",
        "specialization": "Glaucoma & Vitreo-Retinal Surgery",
        "hospital": "Aditya Jyot Eye Hospital",
        "city": "Bengaluru",
        "address": "A-3, Sai Krupa, Link Road, Near SBI, Matunga",
        "phone": "+91-22-2414 4444",
        "experience": "35+ years",
        "qualifications": "MBBS, MS, FAICO, FRCS",
        "rating": "4.8/5",
        "appointment": "www.adityajyoteyehospital.org"
    },
    
    {
        "name": "Dr. R. Kishore Kumar",
        "specialization": "Glaucoma & Cataract",
        "hospital": "Manipal Hospital",
        "city": "Bengaluru",
        "address": "Old Airport Road, Bengaluru",
        "phone": "+91-80-2502 4444",
        "experience": "20+ years",
        "qualifications": "MBBS, MS, DNB",
        "rating": "4.7/5",
        "appointment": "www.manipalhospitals.com"
    },
    {
        "name": "Dr. P. S. Shankar",
        "specialization": "Glaucoma & Ocular Trauma",
        "hospital": "Vikram Hospital",
        "city": "Bengaluru",
        "address": "Miller's Road, Vasanth Nagar",
        "phone": "+91-80-2333 8888",
        "experience": "25+ years",
        "qualifications": "MBBS, MS, FICO",
        "rating": "4.5/5",
        "appointment": "www.vikramhospital.com"
    },
    {
        "name": "Dr. Uday Gadkari",
        "specialization": "Glaucoma & Anterior Segment",
        "hospital": "Gadkari Eye Hospital",
        "city": "Hubballi",
        "address": "Station Road, Hubballi",
        "phone": "+91-836-235 5678",
        "experience": "22+ years",
        "qualifications": "MBBS, MS (Ophthalmology)",
        "rating": "4.6/5",
        "appointment": "gadkarieyehospital.com"
    }
]

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Smart Glaucoma Detector",
    page_icon="👁️",
    layout="wide"
)

# ----------------------------------------------------------
# CSS: animations, glass cards, no visible scrollbars
# ----------------------------------------------------------
st.markdown("""
<style>

/* Hide scrollbars visually but keep scrolling working */
::-webkit-scrollbar { width: 0px; height: 0px; }
* { scrollbar-width: none; }

/* Fade-in animation */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Login box specific styles */
.login-box {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 35px;
    margin: 50px auto;
    max-width: 450px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    animation: fadeIn 0.8s ease-in-out;
    border: 1px solid rgba(255,255,255,0.2);
}

.login-title {
    font-size: 28px;
    font-weight: 900;
    color: #002b5c;
    text-align: center;
    margin-bottom: 8px;
}

.login-subtitle {
    font-size: 14px;
    color: #555;
    text-align: center;
    margin-bottom: 25px;
}

/* Title + subtitle */
.title {
    font-size: 34px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 8px #000;
    margin-top: 10px;
    animation: fadeIn 1s ease-in-out;
}
.subtitle {
    font-size: 14px;
    text-align: center;
    color: #e6e6e6;
    margin-bottom: 10px;
    text-shadow: 0 0 5px #000;
    animation: fadeIn 1.4s ease-in-out;
}

/* Main glass container */
.main-box {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 10px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    animation: fadeIn 0.9s ease-in-out;
}

/* Navigation buttons */
.nav-btn {
    background: linear-gradient(90deg, #0066ff, #00c6ff);
    color: white !important;
    padding: 8px 14px;
    border-radius: 6px;
    border: none;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: 0.25s all ease;
}
.nav-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 0 12px rgba(0,198,255,0.9);
}

/* Section headers */
.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #002b5c;
    margin-bottom: 6px;
}
.subtext {
    font-size: 15px;
    color: #003366;
}

/* Prediction card */
.pred-card {
    background: rgba(255,255,255,0.7);
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.25);
    animation: fadeIn 1s ease-in-out;
}

/* Doctor card styling */
.doctor-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    border-left: 5px solid #0066ff;
    transition: transform 0.3s ease;
}
.doctor-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
.doctor-name {
    font-size: 20px;
    font-weight: 800;
    color: #002b5c;
    margin-bottom: 5px;
}
.doctor-spec {
    font-size: 16px;
    color: #0066ff;
    font-weight: 600;
    margin-bottom: 8px;
}
.doctor-detail {
    font-size: 14px;
    color: #555;
    margin: 3px 0;
}
.doctor-contact {
    background: #e3f2fd;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}

/* Risk colors */
.risk-high { color: red; font-weight: bold; }
.risk-medium { color: orange; font-weight: bold; }
.risk-low { color: green; font-weight: bold; }

/* Login buttons */
.login-btn {
    background: linear-gradient(90deg, #0066ff, #00c6ff);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 14px;
    width: 100%;
    margin-top: 15px;
    margin-bottom: 10px;
    transition: 0.3s;
}
.login-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 0 14px rgba(0,170,255,0.9);
}

/* Buttons (predict) */
.stButton>button {
    background: linear-gradient(90deg, #007BFF, #00BBFF);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 14px;
    font-weight: 700;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 14px rgba(0,170,255,0.9);
}

/* User info in navbar */
.user-info {
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(255,255,255,0.9);
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: #002b5c;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

/* Footer */
.footer {
    text-align: center;
    font-size: 11px;
    color: #f0f0f0;
    text-shadow: 0 0 5px black;
    margin-bottom: 5px;
}

/* Error message styling */
.error-box {
    background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    animation: fadeIn 0.5s ease-in-out;
}

/* Warning message styling */
.warning-box {
    background: linear-gradient(90deg, #ffa726, #ffb74d);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    animation: fadeIn 0.5s ease-in-out;
}

/* Success message styling */
.success-box {
    background: linear-gradient(90deg, #4caf50, #66bb6a);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    animation: fadeIn 0.5s ease-in-out;
}

/* Image validation status */
.validation-pass {
    border: 3px solid #4CAF50;
    border-radius: 10px;
    padding: 5px;
}

.validation-fail {
    border: 3px solid #f44336;
    border-radius: 10px;
    padding: 5px;
}

/* Emergency contact box */
.emergency-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 5px 20px rgba(255,107,107,0.3);
}
.emergency-title {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 10px;
}
.emergency-number {
    font-size: 32px;
    font-weight: 900;
    margin: 10px 0;
}

/* City filter chips */
.city-chip {
    display: inline-block;
    padding: 8px 16px;
    margin: 5px;
    background: #e3f2fd;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.3s;
    font-weight: 500;
}
.city-chip:hover {
    background: #bbdefb;
}
.city-chip.active {
    background: #2196f3;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# INITIALIZE SESSION STATE
# ----------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "show_register" not in st.session_state:
    st.session_state["show_register"] = False
if "last_validated_image" not in st.session_state:
    st.session_state["last_validated_image"] = None
if "validation_result" not in st.session_state:
    st.session_state["validation_result"] = None
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "All"

# ----------------------------------------------------------
# LOGIN/LOGOUT FUNCTIONS
# ----------------------------------------------------------
def login(username, password):
    """Handle user login"""
    success, message = authenticate_user(username, password)
    if success:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        users = load_users()
        st.session_state["role"] = users.get(username, {}).get("role", "user")
        st.session_state["page"] = "home"
        st.success(f"Welcome, {username}!")
        st.rerun()
    else:
        st.error(message)

def logout():
    """Handle user logout"""
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""
    st.session_state["page"] = "login"
    st.session_state["last_validated_image"] = None
    st.session_state["validation_result"] = None
    st.session_state["selected_city"] = "All"
    st.success("Logged out successfully!")
    st.rerun()

# ----------------------------------------------------------
# BACKGROUND STYLE BASED ON AUTHENTICATION
# ----------------------------------------------------------
if not st.session_state["authenticated"]:
    # Login page - plain white background
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Main app - background with image
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: url("app/background.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------
if not st.session_state["authenticated"]:
    # Show login or register form
    if st.session_state["show_register"]:
        # REGISTRATION FORM
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div class='login-title'>📝 Create Account</div>", unsafe_allow_html=True)
                st.markdown("<div class='login-subtitle'>Register for Smart Glaucoma Detector</div>", unsafe_allow_html=True)
                
                with st.form("register_form"):
                    new_username = st.text_input("Username", placeholder="Choose a username")
                    new_password = st.text_input("Password", type="password", placeholder="Create a password")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                    email = st.text_input("Email (optional)", placeholder="your.email@example.com")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        submit_register = st.form_submit_button("Register", use_container_width=True)
                    with col_btn2:
                        back_to_login = st.form_submit_button("Back to Login", use_container_width=True)
                
                if submit_register:
                    if not new_username or not new_password:
                        st.error("Please fill in all required fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = register_user(new_username, new_password, email)
                        if success:
                            st.success(message)
                            st.session_state["show_register"] = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                
                if back_to_login:
                    st.session_state["show_register"] = False
                    st.rerun()
                
                # Demo credentials note
            
    else:
        # LOGIN FORM
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<div class='login-title'>🔐 Smart Glaucoma Detector</div>", unsafe_allow_html=True)
                st.markdown("<div class='login-subtitle'>Academic Use Only • Secure Login Required</div>", unsafe_allow_html=True)
                
                with st.form("login_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        submit_login = st.form_submit_button("Login", use_container_width=True)
                    with col_btn2:
                        register_btn = st.form_submit_button("Register", use_container_width=True)
                
                if submit_login:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        login(username, password)
                
                if register_btn:
                    st.session_state["show_register"] = True
                    st.rerun()
                
                
    
    st.stop()  # Stop execution here if not authenticated

# ----------------------------------------------------------
# MAIN APP (AFTER LOGIN)
# ----------------------------------------------------------
# Show user info in top right
st.markdown(f"""
<div class='user-info'>
👤 {st.session_state['username']} | {st.session_state['role'].upper()}
</div>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='title'>👁️ SMART GLAUCOMA DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning Based Fundus Screening </div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# NAVIGATION - UPDATED WITH DOCTORS OPTION
# ----------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    if st.button("🏠 Home"):
        st.session_state["page"] = "home"
with m2:
    if st.button("ℹ️ About Project"):
        st.session_state["page"] = "about"
with m3:
    if st.button("🧬 Disease Info"):
        st.session_state["page"] = "info"
with m4:
    if st.button("🩺 Prediction"):
        st.session_state["page"] = "predict"
with m5:
    if st.button("👨‍⚕️ Consult Doctors"):
        st.session_state["page"] = "doctors"
with m6:
    if st.button("🚪 Logout"):
        logout()

page = st.session_state["page"]

# ----------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------
if page == "home":
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Welcome to Smart Glaucoma Detector</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtext'>An AI-powered prototype for early glaucoma screening using retinal fundus images.</div>",
        unsafe_allow_html=True,
    )
    st.write(
        """
- Uses **Convolutional Neural Networks (ResNet50, DenseNet121)**
- Classifies eye images into **Glaucoma** / **Non-Glaucoma**
- Helps support doctors with **early detection assistance**
- Built as an **MCA Final Year Project** (Academic & Research use only)

### 👤 User Information
- **Logged in as:** {username}
- **Role:** {role}
- **Access:** Full application access

> **Note:** This system is for academic purposes only. Always consult healthcare professionals for medical diagnosis.
        """.format(username=st.session_state["username"], role=st.session_state["role"])
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# ABOUT PROJECT PAGE
# ----------------------------------------------------------
elif page == "about":
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>About the Project</div>", unsafe_allow_html=True)
    st.write(
        """
### 🎯 Project Title  
**Smart Glaucoma Detector using Deep Learning**

### 🎯 Objective  
To build a deep learning-based system that:
- Analyzes retinal fundus images  
- Automatically detects **glaucoma**  
- Provides a **confidence score** and simple diagnostic summary  

### 🛠 Technologies Used  
- **Python**  
- **PyTorch** (Deep Learning)  
- **Streamlit** (Web UI)  
- **PIL / OpenCV** (Image handling)  

### 📂 Models  
- **Algorithm 1 – ResNet50** (pretrained on ImageNet, fine-tuned)  
- **Algorithm 2 – DenseNet121** (dense connections, good feature extraction)  

### 🔒 Security Features
- User authentication system
- Secure password storage (SHA-256 hashing)
- Role-based access (future implementation)
- Session management

### 🔍 Image Validation
- **Advanced validation** to detect non-fundus images
- **Color pattern analysis** for retinal images
- **Texture analysis** to identify proper fundus structure
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# DISEASE INFO PAGE
# ----------------------------------------------------------
elif page == "info":
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Glaucoma – Disease Information</div>", unsafe_allow_html=True)
    st.write(
        """
Glaucoma is a group of eye conditions that damage the **optic nerve**, often due to high eye pressure.  
If left untreated, glaucoma can cause **permanent vision loss**.

### 🧿 Common Types
- **Open-angle glaucoma** – slow, painless loss of vision  
- **Angle-closure glaucoma** – sudden eye pain, headache, nausea  

### ⚠️ Warning Signs
- Patchy blind spots in side (peripheral) or central vision  
- Severe headache, eye pain  
- Halos around lights  
- Blurred vision  

### 🧪 Diagnostic Tests
- **Tonometry** – measures eye pressure  
- **OCT Scan** – optic nerve and retinal nerve fiber analysis  
- **Visual Field Test** – checks peripheral vision  
- **Gonioscopy** – inspects drainage angle  

### 🩺 Treatment Options
- Prescription **eye drops**  
- **Laser treatment**  
- **Surgery** in advanced cases  

> This project does **not** replace clinical diagnosis. It is only a support tool for learning and research.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# PREDICTION PAGE (INTERACTIVE)
# ----------------------------------------------------------
elif page == "predict":
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI-based Glaucoma Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtext'>Upload a retinal fundus image and select a deep learning model for analysis.</div>",
        unsafe_allow_html=True,
    )

    # Create columns with adjusted ratios and spacing
    left, middle, right = st.columns([1.3, 0.2, 1.5])  # Added middle column for spacing

    with left:
        st.subheader("📤 Upload Fundus Image")
        uploaded = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])
        
        # Show accepted image types
        st.caption("Accepted: JPG, JPEG, PNG | Minimum size: 100x100 pixels")
        
        # Add some vertical spacing
        st.write("")
        st.write("")
        
        st.subheader("⚙️ Select Model")
        model_choice = st.selectbox("Model", ["ResNet50", "DenseNet121"])
        
        # Add more vertical spacing before button
        st.write("")
        st.write("")

        analyze_btn = st.button("🚀 Run AI Analysis", use_container_width=True)
        
        # Add some space at the bottom of the left column
        st.write("")
        st.write("")

    # Middle column (empty for spacing)
    with middle:
        st.write("")  # Just empty space

    with right:
        st.subheader("👁 Image Preview & Validation")
        img = None
        validation_passed = False
        
        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
                
                # Validate the image
                if uploaded != st.session_state.get("last_validated_image"):
                    with st.spinner("🔍 Validating image type..."):
                        validation_passed, validation_message = validate_fundus_image(img)
                        st.session_state["last_validated_image"] = uploaded
                        st.session_state["validation_result"] = (validation_passed, validation_message)
                else:
                    validation_passed, validation_message = st.session_state["validation_result"]
                
                # Display validation result
                if validation_passed:
                    st.markdown("<div class='success-box'>✅ " + validation_message + "</div>", unsafe_allow_html=True)
                    border_class = "validation-pass"
                else:
                    st.markdown("<div class='error-box'>❌ " + validation_message + "</div>", unsafe_allow_html=True)
                    border_class = "validation-fail"
                
                # Display image with appropriate border
                st.markdown(f"<div class='{border_class}'>", unsafe_allow_html=True)
                st.image(img, width=280)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show image details
                with st.expander("📊 Image Details"):
                    st.write(f"**Dimensions:** {img.size[0]} x {img.size[1]} pixels")
                    st.write(f"**Format:** {img.format if hasattr(img, 'format') else 'Unknown'}")
                    st.write(f"**Mode:** {img.mode}")
                    
                    # Calculate and show basic stats
                    img_stats = ImageStat.Stat(img)
                    st.write(f"**Mean brightness:** {img_stats.mean[0]:.1f}")
                    st.write(f"**Standard deviation:** {img_stats.stddev[0]:.1f}")
                    
                    if validation_passed:
                        st.success("✓ Image validated as retinal fundus")
                    else:
                        st.error("✗ Image may not be a retinal fundus")
                        st.info("""
**Expected characteristics of retinal fundus images:**
- Circular/elliptical dark borders
- Reddish color tone (blood vessels)
- Visible optic disc (bright circular area)
- Retinal blood vessels pattern
- Moderate brightness and contrast
                        """)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                img = None
                validation_passed = False
        else:
            st.info("📁 Upload a retinal fundus image to preview and analyze.")
            
            # Show example fundus image characteristics
            with st.expander("👁️ What does a retinal fundus image look like?"):
                st.write("""
**Typical retinal fundus image features:**
1. **Circular/elliptical** field of view
2. **Reddish background** (retinal tissue)
3. **Bright optic disc** (where optic nerve exits)
4. **Branching blood vessels** radiating from optic disc
5. **Dark borders** around the circular field
6. **Macula** (dark spot near center for sharp vision)

**Examples of valid images:**
- Images from retinal cameras (fundus photography)
- Images showing the back of the eye
- Clear view of optic nerve head
                """)

    st.write("---")

    if analyze_btn:
        if uploaded is None:
            st.warning("⚠️ Please upload a fundus image before running analysis.")
        else:
            if not validation_passed:
                st.markdown("<div class='error-box'>", unsafe_allow_html=True)
                st.error("""
❌ **Invalid Image Detected**

The uploaded image does not appear to be a retinal fundus image. 
Please upload a valid retinal fundus image for glaucoma analysis.

**Possible issues:**
1. Image is not of a human eye/retina
2. Wrong image type (face, object, document, etc.)
3. Poor quality or blurry image
4. Incorrect cropping or orientation

**Solution:**
- Upload a clear retinal fundus image
- Ensure the image shows the back of the eye
- Check that optic disc and blood vessels are visible
                """)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show example of what to upload
                with st.expander("📸 See example of valid fundus image"):
                    st.write("""                            
**Key features to look for:**
1. Circular/elliptical field
2. Reddish background color
3. Bright optic disc
4. Visible blood vessels
                    """)
            else:
                with st.spinner("🔬 Running deep learning model..."):
                    model_path = RESNET_MODEL_PATH if model_choice == "ResNet50" else DENSENET_MODEL_PATH
                    model_type = "resnet50" if model_choice == "ResNet50" else "densenet121"

                    if not os.path.exists(model_path):
                        st.error(f"❌ Model file not found: {model_path}")
                    else:
                        model, class_names, device = load_model(model_path, model_type)
                        pred, conf, probs = predict_image(model, device, img, class_names)

                st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
                st.markdown("<div class='success-box'>✅ Image validated successfully as retinal fundus</div>", unsafe_allow_html=True)
                st.write(f"### 👁️ Prediction: **{pred.upper()}**")
                st.write(f"### 🔢 Confidence: **{conf*100:.2f}%**")

                # Risk level display
                risk = conf * 100
                if pred.lower() == "glaucoma":
                    if risk >= 90:
                        color_class = "risk-high"
                        risk_text = "HIGH RISK – Strong signs of glaucoma"
                    elif risk >= 70:
                        color_class = "risk-medium"
                        risk_text = "MODERATE RISK – Possible glaucoma"
                    else:
                        color_class = "risk-low"
                        risk_text = "LOW RISK – Mild suspicion"
                else:
                    color_class = "risk-low"
                    risk_text = "LOW RISK – Appears normal"

                st.markdown(
                    f"<p class='{color_class}'>🩸 Risk Level: {risk_text}</p>",
                    unsafe_allow_html=True,
                )

                # Probabilities (if probs available)
                if probs is not None:
                    st.write("#### 📊 Class Probabilities")
                    try:
                        for cls, p in zip(class_names, probs):
                            st.write(f"{cls}: {p*100:.2f}%")
                            st.progress(float(p))
                    except Exception:
                        pass

                # User info in prediction
                st.write(f"**👤 Analyzed by:** {st.session_state['username']}")
                st.write(f"**📅 Session:** {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Show doctor consultation recommendation
                if pred.lower() == "glaucoma" and risk >= 70:
                    st.markdown("---")
                    st.markdown("### 👨‍⚕️ **Medical Consultation Recommended**")
                    st.warning("""
Based on the AI prediction showing moderate to high risk of glaucoma, 
it is strongly recommended to consult an ophthalmologist for proper diagnosis and treatment.

**Immediate Steps:**
1. Book an appointment with a glaucoma specialist
2. Get comprehensive eye examination
3. Follow medical advice for treatment
4. Regular monitoring as advised by doctor
                    """)
                    
                    # Quick link to doctors page
                    if st.button("📋 Find Glaucoma Specialists in Karnataka"):
                        st.session_state["page"] = "doctors"
                        st.rerun()

                # Detailed explanation
                with st.expander("📘 View Detailed Clinical-style Interpretation"):
                    if pred.lower() == "glaucoma":
                        st.write(
                            """
**Observations (AI-based):**
- Pattern similar to glaucomatous optic nerve  
- Possible elevation in intraocular pressure  
- Retinal nerve fiber loss may be present  

**Recommended Clinical Tests:**
- Optical Coherence Tomography (**OCT**)  
- Visual field test (**Perimetry**)  
- Tonometry (**Eye pressure test**)  
- Gonioscopy  

**Suggested Action (Non-medical advice):**
- Visit an ophthalmologist as soon as possible  
- Do **not** rely only on this system for any treatment decision  
                            """
                        )
                    else:
                        st.write(
                            """
**Observations (AI-based):**
- No strong glaucomatous patterns detected  
- Optic nerve and retina appear similar to healthy samples  

**Suggested Action:**
- Maintain regular eye check-ups (every 6–12 months)  
- If any symptoms appear (eye pain, vision changes), consult a doctor  

Again, this tool is only a **learning prototype**.
                            """
                        )

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# CONSULT DOCTORS PAGE
# ----------------------------------------------------------
elif page == "doctors":
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👨‍⚕️ Consult Glaucoma Specialists in Karnataka</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtext'>List of Top 10 Glaucoma Specialists & Eye Hospitals in Karnataka</div>",
        unsafe_allow_html=True,
    )
    
    # Emergency contact box
    st.markdown("<div class='emergency-box'>", unsafe_allow_html=True)
    st.markdown("<div class='emergency-title'>🚨 Emergency Eye Care</div>", unsafe_allow_html=True)
    st.write("For sudden eye pain, vision loss, or eye injuries:")
    st.markdown("<div class='emergency-number'>📞 108 (Ambulance) or 102 (Eye Emergency)</div>", unsafe_allow_html=True)
    st.write("24/7 Emergency Eye Care available at major hospitals")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # City filter
    st.write("### 📍 Filter by City")
    cities = ["All", "Bengaluru", "Mangaluru", "Hubballi"]
    
    cols = st.columns(len(cities))
    for idx, city in enumerate(cities):
        with cols[idx]:
            if st.button(city, key=f"city_{city}"):
                st.session_state["selected_city"] = city
                st.rerun()
    
    # Display selected city status
    if st.session_state["selected_city"] != "All":
        st.info(f"Showing doctors in: **{st.session_state['selected_city']}**")
    
    st.write("---")
    
    # Display doctors
    filtered_doctors = KARNATAKA_DOCTORS
    if st.session_state["selected_city"] != "All":
        filtered_doctors = [doc for doc in KARNATAKA_DOCTORS if doc["city"] == st.session_state["selected_city"]]
    
    if not filtered_doctors:
        st.warning(f"No doctors found in {st.session_state['selected_city']}. Please select 'All' to see all doctors.")
    else:
        for i, doctor in enumerate(filtered_doctors):
            st.markdown(f"""
            <div class='doctor-card'>
                <div class='doctor-name'>{doctor['name']}</div>
                <div class='doctor-spec'>{doctor['specialization']}</div>
                <div class='doctor-detail'>🏥 <strong>Hospital:</strong> {doctor['hospital']}</div>
                <div class='doctor-detail'>📍 <strong>City:</strong> {doctor['city']}</div>
                <div class='doctor-detail'>🏠 <strong>Address:</strong> {doctor['address']}</div>
                <div class='doctor-detail'>📅 <strong>Experience:</strong> {doctor['experience']}</div>
                <div class='doctor-detail'>🎓 <strong>Qualifications:</strong> {doctor['qualifications']}</div>
                <div class='doctor-detail'>⭐ <strong>Rating:</strong> {doctor['rating']}</div>
                <div class='doctor-contact'>
                    📞 <strong>Phone:</strong> {doctor['phone']}<br>
                    🌐 <strong>Appointment:</strong> {doctor['appointment']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Important notes section
    st.write("---")
    st.markdown("### 📝 Important Information")
    
    col_note1, col_note2 = st.columns(2)
    
    with col_note1:
        st.markdown("""
        **🔔 Booking Tips:**
        1. Call during hospital hours (9 AM - 5 PM)
        2. Carry previous medical records
        3. Book follow-up appointments
        4. Confirm insurance coverage
        
        **💼 What to Bring:**
        - Previous eye test reports
        - Current medications list
        - ID proof and insurance
        - Referral letter if any
        """)
    
    with col_note2:
        st.markdown("""
        **⚠️ Medical Disclaimer:**
        - This list is for informational purposes only
        - Always verify credentials independently
        - Consult your regular doctor first
        - Emergency cases: Visit nearest hospital
        
        **🏥 Major Eye Hospitals:**
        - Narayana Nethralaya, Bengaluru
        - Minto Eye Hospital, Bengaluru
        - Manipal Hospital, Bengaluru
        - Shetty Eye Centre, Mangaluru
        """)
    
    # Quick actions
    st.write("---")
    st.markdown("### ⚡ Quick Actions")
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        if st.button("📅 Book Nearest Appointment", use_container_width=True):
            st.info("Please call the hospital numbers listed above for appointments")
    
    with col_act2:
        if st.button("🗺️ View on Map", use_container_width=True):
            st.info("Map integration can be added. For now, use addresses provided.")
    
    with col_act3:
        if st.button("📋 Download List", use_container_width=True):
            # Create downloadable text
            doctor_text = "Glaucoma Specialists in Karnataka\n"
            doctor_text += "=" * 40 + "\n\n"
            for doc in KARNATAKA_DOCTORS:
                doctor_text += f"Name: {doc['name']}\n"
                doctor_text += f"Specialization: {doc['specialization']}\n"
                doctor_text += f"Hospital: {doc['hospital']}\n"
                doctor_text += f"City: {doc['city']}\n"
                doctor_text += f"Address: {doc['address']}\n"
                doctor_text += f"Phone: {doc['phone']}\n"
                doctor_text += f"Experience: {doc['experience']}\n"
                doctor_text += f"Appointment: {doc['appointment']}\n"
                doctor_text += "-" * 40 + "\n"
            
            st.download_button(
                label="⬇️ Download Doctor List",
                data=doctor_text,
                file_name="glaucoma_specialists_karnataka.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown(
    "<div class='footer'>Smart Glaucoma Detector • MCA Project • UBDTCE Davangere • Academic Use Only</div>",
    unsafe_allow_html=True,
)