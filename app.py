import streamlit as st
import pandas as pd

st.header("Seattle Gas Price Prediction App")

# load data

# load model
darnn = DARNN(N=data.shape[1]-1, M=64, P=64,
              T=8, device='cpu')