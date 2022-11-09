import streamlit as st
from io import StringIO
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


uploaded_file = st.file_uploader("Choose Your CSV file")

if uploaded_file is not None:

    bytes_data = uploaded_file.getvalue()

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    string_data = stringio.read()

    df = pd.read_csv(uploaded_file)


    def gompertz(x, a, b, c):
        y = a * np.exp(-b * np.exp(-c * x))
        return y


    def four_pl(x, A, B, C, D):
        y = ((A - D) / (1.0 + ((x / C) ** B)) + D)
        return y


    def five_pl(x, A, B, C, D, E):
        y = D + (A - D) / (np.power((1 + np.power((x / C), B)), E))
        return y


    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y


    def power_law(x, a, b):
        return a * np.power(x, b)


    def Average(lst):
        return sum(lst) / len(lst)


    # genre = st.dataframe(df)
    filtered = st.multiselect("Filter columns", options=list(df.columns), default=list(df.columns))

    st.write(df[filtered])
    if st.sidebar.checkbox("Show Raw Data for X axis ", False):
        dosage = df[filtered[0]]
        st.subheader("From the CSV file first column will be selected as the value for X Axis")
        st.write(dosage)

    if st.sidebar.checkbox("Show Raw Data for Y axis", False):
        efficacy_val = df[filtered[1]]
        st.subheader("From the CSV file second column will be selected as the value for Y Axis")
        st.write(efficacy_val)
    if st.sidebar.checkbox("Show Raw Data ", False):
        st.subheader("Uploaded File 📁")
        st.write(df)

    st.sidebar.subheader("Choose Growth Function")
    functions = st.sidebar.selectbox("Functions",
                                     ("Gompertz Function", "Four Parameter Logistic Function",
                                      "Five Parameter Logistic Function",
                                      "Sigmoid Function", "Power Law Function"))

    if functions == "Gompertz Function":
        st.sidebar.subheader('Initial Parameters')
        xdata = df[filtered[0]]
        ydata = df[filtered[1]]
        max_dosage = int(max(xdata))
        min_dosage = int(min(xdata))
        max_eff = int(max(ydata))
        min_eff = int(min(ydata))
        A = st.sidebar.slider("Initial Parameter for A", max_dosage, min_dosage)
        D = st.sidebar.slider("Initial Parameter for B", max_eff, min_eff)
        method = st.sidebar.radio("Method", {'lm', 'trf', 'dogbox'}, key='method')
        x = xdata
        B = (D - A) / (np.amax(x) - np.amin(x))
        popt, pcov = curve_fit(gompertz, xdata, ydata, p0=[max_eff, round(Average(x)), 1], method=method,
                               maxfev=900000000, gtol=1e-10)
        xlist = np.linspace(0, max_dosage, 1500)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        y_fitted1 = gompertz(xdata, *popt)
        rscore_weith1 = r2_score(y_fitted1, ydata)

        if st.sidebar.button("Button", key='functions'):
            st.write("R2 Score: ", rscore_weith1)
            st.write("Calculated A Parameter: ", A)
            st.write("Calculated B Parameter: ", B)

            plt.plot(xlist, gompertz(xlist, *popt), 'black', label='Gompertz R2 Value=')
            st.write("Gompertz Function : a * np.exp(-b * np.exp(-c * x))")
            st.pyplot(fig)

    if functions == "Four Parameter Logistic Function":
        st.sidebar.subheader('Initial Parameters')
        xdata = df[filtered[0]]
        ydata = df[filtered[1]]
        x = xdata
        max_dosage = int(max(xdata))
        min_dosage = int(min(xdata))
        max_eff = int(max(ydata))
        min_eff = int(min(ydata))
        A = st.sidebar.slider("Initial Parameter for A", max_dosage, min_dosage)
        D = st.sidebar.slider("Initial Parameter for B", max_eff, min_eff)
        B = (D - A) / (max_dosage - min_dosage)
        C = st.sidebar.slider("Initial Parameter for C", 0, 10, step=1, key='C')
        method = st.sidebar.radio("Method", {'lm', 'trf', 'dogbox'}, key='method')
        popt, pcov = curve_fit(four_pl, xdata, ydata, p0=[A, B, C, D], method=method,
                               maxfev=900000000, gtol=1e-10)
        xlist = np.linspace(0, max_dosage, 1500)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        y_fitted1 = four_pl(xdata, *popt)
        rscore_weith1 = r2_score(y_fitted1, ydata)
        if st.sidebar.button("Button", key='functions'):
            st.write("R2 Score: ", rscore_weith1)
            st.write("Calculated A Parameter: ", A)
            st.write("Calculated B Parameter: ", B)
            st.write("Calculated C Parameter: ", C)
            st.write("Calculated D Parameter: ", D)
            plt.plot(xlist, four_pl(xlist, *popt), 'black', label='Four Parameter Logistic Functions R2 Value=')
            st.write("Four Parameter Logistic Function")
            st.pyplot(fig)

    if functions == "Five Parameter Logistic Function":
        st.sidebar.subheader('Initial Parameters')
        xdata = df[filtered[0]]
        ydata = df[filtered[1]]
        x = xdata
        max_dosage = int(max(xdata))
        min_dosage = int(min(xdata))
        max_eff = int(max(ydata))
        min_eff = int(min(ydata))
        A = st.sidebar.slider("Initial Parameter for A", max_dosage, min_dosage)
        D = st.sidebar.slider("Initial Parameter for B", max_eff, min_eff)
        B = (D - A) / (max_dosage - min_dosage)
        C=(max_dosage-min_dosage)/2
        E = st.sidebar.slider("Initial Parameter for D", 0, 10, step=1, key='E')
        method = st.sidebar.radio("Method", {'lm', 'trf', 'dogbox'}, key='method')
        popt, pcov = curve_fit(five_pl, xdata, ydata, p0=[A, B, C, D, E], method=method,
                               maxfev=900000000, gtol=1e-10)
        xlist = np.linspace(0, max_dosage, 1500)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        y_fitted1 = five_pl(xdata, *popt)
        rscore_weith1 = r2_score(y_fitted1, ydata)
        if st.sidebar.button("Button", key='functions'):
            st.write("R2 Score: ", rscore_weith1)
            st.write("Calculated A Parameter: ", A)
            st.write("Calculated B Parameter: ", B)
            st.write("Calculated C Parameter: ", C)
            st.write("Calculated D Parameter: ", D)
            st.write("Calculated E Parameter: ", E)
            plt.plot(xlist, five_pl(xlist, *popt), 'black', label='Five Parameter Logistic Functions R2 Value=')
            st.write("Four Parameter Logistic Function")
            st.pyplot(fig)
    if functions == "Sigmoid Function":
        st.sidebar.subheader('Initial Parameters')
        xdata = df[filtered[0]]
        ydata = df[filtered[1]]
        x = xdata
        max_dosage = int(max(xdata))
        min_dosage = int(min(xdata))
        max_eff = int(max(ydata))
        min_eff = int(min(ydata))
        A = st.sidebar.slider("Initial Parameter for A", max_dosage, min_dosage)
        B = st.sidebar.slider("Initial Parameter for B", max_eff, min_eff)
        C=st.sidebar.slider("Initial Parameter for C", 0, 10, step=1, key='C')
        D = st.sidebar.slider("Initial Parameter for D", 0, 10, step=1, key='D')
        method = st.sidebar.radio("Method", {'lm', 'trf', 'dogbox'}, key='method')
        popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=[A, B, C, D], method=method,
                               maxfev=900000000, gtol=1e-10)
        xlist = np.linspace(0, max_dosage, 1500)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        y_fitted1 = sigmoid(xdata, *popt)
        rscore_weith1 = r2_score(y_fitted1, ydata)
        if st.sidebar.button("Button", key='functions'):
            st.write("R2 Score: ", rscore_weith1)
            st.write("Calculated A Parameter: ", A)
            st.write("Calculated B Parameter: ", B)
            st.write("Calculated C Parameter: ", C)
            st.write("Calculated D Parameter: ", D)
            plt.plot(xlist, sigmoid(xlist, *popt), 'black', label='Sigmoid Functions R2 Value=')
            st.write("Sigmoid Function")
            st.pyplot(fig)
    if functions == "Power Law Function":
        st.sidebar.subheader('Initial Parameters')
        xdata = df[filtered[0]]
        ydata = df[filtered[1]]
        x = xdata
        max_dosage = int(max(xdata))
        min_dosage = int(min(xdata))
        max_eff = int(max(ydata))
        min_eff = int(min(ydata))
        A = st.sidebar.slider("Initial Parameter for A", max_dosage, min_dosage)
        B = st.sidebar.slider("Initial Parameter for B", max_eff, min_eff)
        method = st.sidebar.radio("Method", {'lm', 'trf', 'dogbox'}, key='method')
        popt, pcov = curve_fit(power_law, xdata, ydata, p0=[A, B], method=method,
                               maxfev=900000000, gtol=1e-10)
        xlist = np.linspace(0, max_dosage, 1500)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        y_fitted1 = power_law(xdata, *popt)
        rscore_weith1 = r2_score(y_fitted1, ydata)
        if st.sidebar.button("Button", key='functions'):
            st.write("R2 Score: ", rscore_weith1)
            st.write("Calculated A Parameter: ", A)
            st.write("Calculated B Parameter: ", B)
            plt.plot(xlist, power_law(xlist, *popt), 'black', label='Power Law Functions R2 Value=')
            st.write("Sigmoid Function")
            st.pyplot(fig)

