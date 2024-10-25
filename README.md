# Tubitak


### EDA
- Ay bazında açılış ve kapanış fiyatlarını görselleştir
- Periyodu tekrardan görselleştir

### Denenecek Teknikler
- Mimax scaler
- Prophet
- Arima
- Autogluon
- Pycaret

### Periyod
- Tüm data Train
- 6 Ay trainx
- Periyodik modelin eğitilmesi gerekli

### MA
    # moving average
    df['50-Day MA'] = df['Close'].rolling(window=50).mean()
    df['200-Day MA'] = df['Close'].rolling(window=200).mean()
