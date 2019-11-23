from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Layout,VBox
import ipywidgets as widgets
import matplotlib.pyplot as plt
import predict_dgn
import numpy as np
 

class PlotMono:
    def __init__(self):
        #fix the widget width to fit long descriptions
        style = {'description_width': 'initial'}
        
        
        self.breast_radius = widgets.BoundedFloatText(
            min=6,
            max=12.0,
            step=0.1,
            value=10,
            description='Breast radius (cm):',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.breast_thickness = widgets.BoundedFloatText(
            min=2,
            max=11.0,
            step=0.1,
            value=5,
            description='Breast Thickness (cm):',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.gland = widgets.BoundedFloatText(
            min=0,
            max=1.0,
            step=0.001,
            value=0.2,
            description='Glandularity',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.skin = widgets.BoundedFloatText(
            min=0,
            max=5.0,
            step=0.01,
            value=1.45,
            description='Skin Thickness (mm)',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.adipose = widgets.BoundedFloatText(
            min=0,
            max=5.0,
            step=0.01,
            value=0,
            description='Adipose Thickness (mm)',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.ccpd = widgets.BoundedFloatText(
            min=0,
            max=40.0,
            step=0.1,
            value=0,
            description='CPCD',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.bplot = widgets.ToggleButton(
            value=False,
            description='Plot',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Generate Plot',
            icon='check'
        )

        self.explot = widgets.Checkbox(
                    value=False,
                    description='Export Data',
                    disabled=False
                    )

        

    #define code to plot
    def plot_spectrum(self, breast_radius, breast_thickness, gland, skin, adipose, ccpd, bplot, explot):

        if bplot:
            energy_vec = np.arange(8.25, 49.25, 0.5)
            dgn_vec = []
            e_dgn_vec = []
            for energy in energy_vec:
                mgd, error_mgd = predict_dgn.predict_mgd_mono(energy, breast_radius, breast_thickness, gland, skin, adipose)
                kair, error_kair = predict_dgn.predict_kerma_mono(energy, breast_thickness, ccpd)
                dgn = mgd/kair
                e_dgn = dgn*np.sqrt((error_mgd/mgd)**2 + (error_kair/kair)**2)
                dgn_vec.append(dgn)
                e_dgn_vec.append(e_dgn)

            plt.figure(figsize=(6,4))
            plt.errorbar(energy_vec, dgn_vec, yerr=e_dgn_vec)
            plt.xlabel('Energy (keV)')
            plt.ylabel('DgN (mGy/mGy)')
            plt.tight_layout()
            plt.show()

            self.bplot.value = False
            if explot:
                self.exportdata(energy_vec, dgn_vec, e_dgn_vec)

        return
    
    #wrap with interaction
    def interaction(self):
        widget= interactive(self.plot_spectrum, breast_radius= self.breast_radius, breast_thickness = self.breast_thickness, gland=self.gland, 
                    skin=self.skin,adipose=self.adipose, ccpd = self.ccpd,  bplot= self.bplot, explot=self.explot)
        return widget


    def exportdata(self, energy, dgn, edgn):
        file = open('DgNMono.txt', 'w')
        file.write('###Monoenergetic DgN Data generated with ANN####\n')
        file.write('#Configuration\n')
        file.write('#Breast radius (cm) = '+str(self.breast_radius.value)+'\n')
        file.write('#Breast Thickness = '+str(self.breast_radius.value)+'\n')
        file.write('#Breast Glandularity = '+str(self.gland.value)+'\n')
        file.write('#Skin Thickness (mm) = '+str(self.skin.value)+'\n')
        file.write('#Adipose Thickness (mm) = '+str(self.adipose.value)+'\n')
        file.write('#CPCD = '+str(self.ccpd.value)+'\n')
        file.write('#Energy(keV)\tDgN\tUncertainty\n')
        for i in range(len(energy)):
            #file.write(' '+str(potential[i])+'\t'+str(np.round(hvl[i],4))+'\t'+str(np.round(dgn[i],5))+'\t'+str(np.round(edgn[i],5))+'\n')
            file.write(f' {energy[i]:5.2f}\t\t{dgn[i]:7.5f}\t{edgn[i]:7.5f}\n')

        file.close()
    
