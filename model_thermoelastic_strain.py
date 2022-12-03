#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy
from obspy.signal.util import smooth
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime as UTC
import matplotlib.dates as dates

#--- jwilgus@unm.edu, Dec. 2022 ---#
#--- script to model thermoelastic variations using equations fom Tsai, 2011

#--- import weather data
f = open('temp_ANMO_2012_2022_dmean.txt', 'r') # mean daily temp [C]
temp_dates = []
temp_vals_ANMO = []

for idx,line in enumerate(f):
    line = line.rstrip()
    line = line.lstrip()
    line = line.split(',')
    temp_dates.append(line[0][0:10].strip('"'))
    temp_vals_ANMO.append(float(line[1].strip('"')))
f.close()

#--- place list info/data into pandas data structure
df_model = pd.DataFrame({'time': pd.to_datetime(temp_dates), 'temp': temp_vals_ANMO},
                      columns=['time','temp']).sort_values('time')

#--- index dates of interest
date_low   = '2013-01-01'
date_hi    = '2017-01-01'
data_QCidx =  ((df_model['time'] >= date_low) &
               (df_model['time'] <= date_hi))
data_QCint = df_model.index[data_QCidx].astype(object) #integers for indexing

# --- gps and dv/v from thermoelastic strain --- #
# Equations from Tsai, 2011
# eq. 5   -->  A(t)  = (1+v)/(1-v)*k*alpha_th*T0*sqrt(K/w)*exp(pi/4-w*delta_t)*cos[w*(t-delta_t)]
# eq. 11b -->  Ux    = -(A_t/k)*cos(k*x)*exp(-k*y)*(2*(1-v)-(k*y))
# eq. 17: -->  dv/v  = (m/mu)*A_t*exp(-ky)*sin(kx)(1-2*v)

# Variables
m_mu= -10000        #m  --> Murnaghan constant #mu --> shear modulus
k=(2*np.pi)/20000   #k  --> horizontal wavenumber -- wavelength of surface temperture variation
x= 1                #x  --> horizontal position
y= 0.5                #y  --> depth [m] --> 0 close to the surface a
#t = 1              #t  --> time
v=0.3               #v  --> Poisson's ratio
alpha_th=10**-5     #alpha_th --> coefficient of thermal expansion
T0=15               #T0  --> half annual peak-to-peak mean surface temperature variation
K=10**-6            #K   --> thermal diffusivity m^2/s
w=2*10**-7          #w   --> frequency /s
per=(2*np.pi)/w     #per --> tau=2pi/w, period
delta_t=(y/2)*np.sqrt(per/(np.pi*K))+(per/8) #delta_t --> lag between surface temp and strain
#delta_t=y/np.sqrt(2*w*K)+(np.pi/(4*w))

#-preallocate dataframe
df_model['doy'] = df_model['time'].dt.dayofyear #start_day
df_model['strain'] = np.nan
df_model['Ux'] = np.nan
df_model['dvv'] = np.nan
df_model['Ux_Tsai'] = np.nan
df_model['dvv_Tsai'] = np.nan

for idx in df_model.index:

    # time --> day of year in seconds
    t = df_model['doy'][idx]*86400

    # model strain amplitude Eq. 5
    A_t = (1+v)/(1-v)*k*alpha_th*T0*np.sqrt(K/w)*np.exp(np.pi/4-(w*delta_t))*np.cos(w*(t-delta_t))
    df_model['strain'][idx] = A_t

    # gps displacements, Ux, from modeled strain Eq. 11b
    # --> approaches 0 as y approaches infinity
    df_model['Ux'][idx] = -(A_t/k)*np.cos(k*x)*np.exp(-k*y)*(2*(1-v)-(k*y))*1000 # convert to mm

    # thermoelastic dv/v eq. 18
    # --> where horizontal position (x) is largest -->remove sin term that contains x
    df_model['dvv'][idx] = m_mu*A_t*np.exp(-k*y)*(1-2*v) #*np.sin(k*x)

    # examples from Tsai, 2011
    # Fig. 3b, Eq. 15, example using values from Prawirodirdjo et al., 2006
    # --> T0= 10°C,k=2p/(20 km),yb= 0.5 m
    df_model['Ux_Tsai'][idx] = -0.5*np.cos(k*x)*np.cos(2*np.pi*(t-(55*86400))/per) #np.cos(k*x) --> -1 ?
    # Fig. 3c, Eq. 18, using same values as Eq. 15
    df_model['dvv_Tsai'][idx] = 4*10**-8*m_mu*np.cos(((2*np.pi)*(t-(55*86400)))/per)*np.exp(-k*y) #*np.sin(k*x)

print(df_model)
df_model.info

### ------------------------------------------------------------------------- ###
# plot modeled results against temperature and compare with figure 3 in Tsai, 2011
fig, ax = plt.subplots(4,1,figsize=(10,9))

# --- temp
ax[0].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['temp'][data_QCint],
                               '-',color='orange',label='Temperature ANMO')
ax[0].autoscale(enable=True, axis='both', tight=True)
ax[0].set_xlim(pd.Timestamp(date_low), pd.Timestamp(date_hi))
ax[0].set_ylabel('Temp [C]')
ax[0].legend(loc='upper left')

# --- modeled strain (A_t)
ax[1].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['strain'][data_QCint],
                               'k-',label='Strain A(t)')
ax[1].autoscale(enable=True, axis='both', tight=True)
ax[1].set_xlim(pd.Timestamp(date_low), pd.Timestamp(date_hi))
ax[1].set_ylabel('strain [-]')
ax[1].legend(loc='upper left', ncol = 2)

# --- modeled ground displacement (Ux)
ax[2].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['Ux_Tsai'][data_QCint],
                               'k-',label='Tsai Fig. 3b')
ax[2].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['Ux'][data_QCint],
                               'b--',label='Ux ANMO')
ax[2].autoscale(enable=True, axis='both', tight=True)
ax[2].set_ylim(-2, 2)
ax[2].set_xlim(pd.Timestamp(date_low), pd.Timestamp(date_hi))
ax[2].set_ylabel('Ux position [mm]')
ax[2].legend(loc='upper left', ncol = 2)

# --- modeled dv/v
ax[3].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['dvv_Tsai'][data_QCint]*100*10,
                               'k-',label='Tsai Fig. 3c')
ax[3].plot_date(dates.date2num(df_model['time'][data_QCint]),df_model['dvv'][data_QCint]*100*10,
                               'g--',label='ANMO')
ax[3].autoscale(enable=True, axis='both', tight=True)
ax[3].set_ylim(-1.2, 1.2)
ax[3].set_xlim(pd.Timestamp(date_low), pd.Timestamp(date_hi))
ax[3].set_ylabel('dv/v [0.1%]')
ax[3].legend(loc='upper left', ncol = 2)

plt.show()
plt.close()
