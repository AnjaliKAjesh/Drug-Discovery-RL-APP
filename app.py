
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from environment import DrugMDP, fragments
from dqn_model import DQN
from utils import encode_smiles

# Setup
st.set_page_config(page_title='RL Molecule Builder', layout='wide')
st.title('💊 Reinforcement Learning-Based Molecule Generator')

# Initialize environment
if 'env' not in st.session_state:
    st.session_state.env = DrugMDP(max_steps=10)
    st.session_state.state = st.session_state.env.reset()

env = st.session_state.env
smiles = env.smiles
mol = Chem.MolFromSmiles(smiles)

# Display current molecule
st.subheader('Current Molecule')
st.image(Draw.MolToImage(mol), caption=smiles)

# Load trained model (if available)
try:
    model = DQN(state_size=64, action_size=len(fragments))
    model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cpu')))
    model.eval()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning("Pretrained model not found. Using manual fragment selection.")

# Fragment selection
st.subheader('Fragment Attachment')
col1, col2 = st.columns(2)

if model_loaded:
    state_tensor = torch.tensor([encode_smiles(smiles)], dtype=torch.float32)
    q_vals = model(state_tensor).detach().numpy().flatten()
    best_action = int(q_vals.argmax())
    with col1:
        st.write("💡 Model recommends:")
        st.code(fragments[best_action])
    action = best_action
else:
    with col1:
        action_label = st.selectbox("Select Fragment", fragments)
        action = fragments.index(action_label)

# Apply action
with col2:
    if st.button('Apply Fragment'):
        new_state, reward, done = env.step(action)
        st.experimental_rerun()

if st.button('Reset Molecule'):
    st.session_state.env = DrugMDP(max_steps=10)
    st.session_state.state = st.session_state.env.reset()
    st.experimental_rerun()
