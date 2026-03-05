# routes/auth_routes.py
from flask import Blueprint, request, render_template, redirect, url_for, flash, session, current_app
import firebase_admin
from firebase_admin import auth as admin_auth, firestore, credentials
from datetime import datetime
import requests

# Initialize Firebase Admin SDK (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate('firebase-adminsdk.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()
auth_blueprint = Blueprint('auth', __name__)

# --- Helpers ---------------------------------------------------------------

def firebase_password_login(email: str, password: str):
    """
    Use Firebase Identity Toolkit REST API to verify email/password.
    Returns dict with idToken, localId (uid), email on success; raises on failure.
    """
    api_key = current_app.config.get('FIREBASE_WEB_API_KEY')
    if not api_key:
        raise RuntimeError("FIREBASE_WEB_API_KEY missing in config.py")

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# --- Routes ---------------------------------------------------------------

@auth_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        username = request.form['username'].strip()

        try:
            # 1) Create user in Firebase Authentication
            user_record = admin_auth.create_user(email=email, password=password)

            # 2) Save profile/role in Firestore (doc id == uid)
            user_data = {
                'username': username,
                'email': email,
                'role': 'user',                 # default role
                'created_at': datetime.utcnow()
            }
            db.collection('users').document(user_record.uid).set(user_data)

            flash('Registration successful! You can now login.', 'success')
            return redirect(url_for('auth.login'))

        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'error')

    return render_template('register.html')


@auth_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        try:
            # 1) Verify credentials with Firebase Auth
            auth_res = firebase_password_login(email, password)
            uid = auth_res['localId']

            # 2) Get role/profile from Firestore
            user_doc = db.collection('users').document(uid).get()
            if not user_doc.exists:
                # If user exists in Auth but not in Firestore, create a minimal doc
                db.collection('users').document(uid).set({
                    'email': email, 'username': email.split('@')[0],
                    'role': 'user', 'created_at': datetime.utcnow()
                })
                role = 'user'
            else:
                role = (user_doc.to_dict() or {}).get('role', 'user')

            # 3) Store session
            session['user_id'] = uid
            session['email'] = email
            session['role'] = role

            # 4) Route by role
            if role == 'admin':
                return redirect(url_for('main.admin_dashboard'))
            return redirect(url_for('main.user_dashboard'))

        except requests.HTTPError as e:
            msg = e.response.json().get('error', {}).get('message', 'Login failed')
            flash(f'Login failed: {msg}', 'error')
        except Exception as e:
            flash(f'Login error: {str(e)}', 'error')

    return render_template('login.html')


@auth_blueprint.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('auth.login'))
