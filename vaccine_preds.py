import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class VaccinePredictor:

    def __init__(self,):

        self.data = pd.read_csv('2021-05-25_trends.csv')
        # Pre-processing the self.data
        self.data = self.data[self.data['Location']=='US']
        self.data = self.data[self.data.notna()]
        self.data = self.data[['Date', 'Administered_7_Day_Rolling_Average']]
        self.data = self.data.dropna()
        self.data = self.data.set_index('Date')
        self.data.index = pd.to_datetime(self.data.index)
        # Fit a Gausssian Curve
        self.gaussian_model, cov = optimize.curve_fit(self.gaussian_f,
            xdata=np.arange(len(self.data['Administered_7_Day_Rolling_Average'])), 
            ydata=self.data['Administered_7_Day_Rolling_Average'].values)

        print('Gaussian fit successfull with parameters:')
        print(self.gaussian_model)

    def gaussian_f(self,X, a, b, c):
        y = a * np.exp(-0.5 * ((X-b)/c)**2)
        return y

    # Plot parametric fitting.
    def utils_plot_parametric(self,data, zoom=30, figsize=(15,5)):
        ## interval
        self.data["residuals"] = self.data["ts"] - self.data["model"]
        self.data["conf_int_low"] = self.data["forecast"] - 1.96*self.data["residuals"].std()
        self.data["conf_int_up"] = self.data["forecast"] + 1.96*self.data["residuals"].std()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        ## entire series
        self.data["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")
        self.data["model"].plot(ax=ax[0], color="green")
        self.data["forecast"].plot(ax=ax[0], grid=True, color="red")
        ax[0].fill_between(x=self.data.index, y1=self.data['conf_int_low'], y2=self.data['conf_int_up'], color='b', alpha=0.3)
    
        ## focus on last
        first_idx = self.data[pd.notnull(self.data["forecast"])].index[0]
        first_loc = self.data.index.tolist().index(first_idx)
        zoom_idx = self.data.index[first_loc-zoom]
        self.data.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 
                                    title="Zoom on the last "+str(zoom)+" observations")
        self.data.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
        self.data.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
        ax[1].fill_between(x=self.data.loc[zoom_idx:].index, y1=self.data.loc[zoom_idx:]['conf_int_low'], 
                        y2=self.data.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
        plt.show()
        return self.data[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]

    def forecast_curve(self,ts, f, model, pred_ahead=None, freq="D", zoom=30, figsize=(15,5)):
        '''
        Forecast unknown future.
        :parameter
            :param ts: pandas series
            :param f: function
            :param model: list of optim params
            :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
            :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
            :param zoom: for plotting
        '''
        ## fit
        X = np.arange(len(ts))
        fitted = f(X, model[0], model[1], model[2])
        self.data = ts.to_frame(name="ts")
        self.data["model"] = fitted
        
        ## index
        index = pd.date_range(start='2021-05-25',periods=pred_ahead,freq=freq)
        index = index[1:]
        ## forecast
        Xnew = np.arange(len(ts)+1, len(ts)+1+len(index))
        preds = f(Xnew, model[0], model[1], model[2])
        self.data = self.data.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
        
        ## plot
        self.utils_plot_parametric(self.data, zoom=zoom)
        return self.data

    def write_forecast_csv(self,pred_ahead=120):

        preds = self.forecast_curve(self.data["Administered_7_Day_Rolling_Average"], self.gaussian_f, self.gaussian_model, 
            pred_ahead=pred_ahead, freq="D", zoom=7)
        
        all_df = pd.concat([preds['model'].dropna(),preds['forecast'].dropna()])
        all_df = pd.DataFrame(all_df)
        all_df.columns = ['Predictions']
        all_df.index.names = ['Date']
        all_df.to_csv('vaccine_forecasts.csv')

if __name__ == "__main__":

    PredObj = VaccinePredictor()
    PredObj.write_forecast_csv()