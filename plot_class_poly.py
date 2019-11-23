from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Layout,VBox
import ipywidgets as widgets
import matplotlib.pyplot as plt
import predict_dgn
import numpy as np

class PDgNp:
    def __init__(self):
        #fix the widget width to fit long descriptions
        style = {'description_width': 'initial'}
        
        #define widgets
        self.anode = widgets.Dropdown(
            options=['Mo', 'Rh', 'W'],
            value='Mo',
            description='Anode*:',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        #define widgets
        self.filter = widgets.Dropdown(
            options=['Ag', 'Al', 'Cu', 'Mo', 'Pd', 'Rh', 'Sn', 'Ti'],
            value='Mo',
            description='Filter:',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.filter_thickness = widgets.BoundedFloatText(
            min=0,
            max=10.0,
            step=0.001,
            value=0.03,
            description='Filter thickness (mm):',
            disabled=False,
            style = {'description_width': 'initial'}
        )

        self.pmma_thickness = widgets.BoundedFloatText(
            min=0,
            max=10.0,
            step=0.001,
            value=0,
            description='PMMA thickness (mm):',
            disabled=False,
            style = {'description_width': 'initial'}
        )

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

        self.spectrum_ref = widgets.Label(
            value='*Unfiltered x-ray spectra from Hernandez et al. (2017)',
            disabled=True,
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

##        self.df_anode_filter = df
##        
##        #update widget value
##        def update_filter_thickness(*args):
##            if self.anode_filter.value == 'Mo/Mo':
##                self.filter_thickness.value = 0.03
##            elif self.anode_filter.value == 'W/Rh':
##                self.filter_thickness.value = 0.05
##            elif self.anode_filter.value == 'W/Ag':
##                self.filter_thickness.value = 0.05
##            
##        self.anode_filter.observe(update_filter_thickness, 'value')


    #define code to plot
    def plot_spectrum(self, anode,Filter, pmma, thick_filter, radius, thick, gland, skin, adipose, ccpd, spectrum_ref, bplot, explot):

        if bplot:
            potential = np.arange(20,50,1)
            vdgn = []
            vedgn = []
            vhvl = []
            for pot in potential:
                dgn, edgn, hvl = predict_dgn.predict_dgn_poli(anode.lower(), Filter, thick_filter, pot, radius, thick, gland, skin, adipose, ccpd, pmma)
                vdgn.append(dgn)
                vedgn.append(edgn)
                vhvl.append(hvl)

            plt.figure(figsize=(6,4))

            plt.errorbar(potential, vdgn, yerr=vedgn)

            plt.xlabel('Potential (kV)')
            plt.ylabel('DgN (mGy/mGy)')
            plt.title(self.anode.value+'/'+self.filter.value)
            plt.tight_layout()

            plt.show()
            
            self.bplot.value = False

            if explot:
                self.exportdata(potential, vhvl, vdgn, vedgn)

        return
    
    #wrap with interaction
    def interaction(self):
        widget= interactive(self.plot_spectrum, anode = self.anode, Filter=self.filter, pmma=self.pmma_thickness,  thick_filter = self.filter_thickness, 
                    radius= self.breast_radius, thick = self.breast_thickness, gland=self.gland, 
                    skin=self.skin,adipose=self.adipose, ccpd = self.ccpd, spectrum_ref = self.spectrum_ref, bplot= self.bplot, explot=self.explot)
        return widget



    def exportdata(self, potential, hvl, dgn, edgn):
        file = open('DgNPoly.txt', 'w')
        file.write('###Polyenergetic DgN Data generated with ANN####\n')
        file.write('#Configuration\n')
        file.write('#Anode = '+str(self.anode.value)+'\n')
        file.write('#Filter = '+str(self.filter.value)+'\n')
        file.write('#Filter Thickness (mm) = '+str(self.filter_thickness.value)+'\n')
        file.write('#Additional PMMA Filtration (mm) = '+str(self.pmma_thickness.value)+'\n')
        file.write('#Breast radius (cm) = '+str(self.breast_radius.value)+'\n')
        file.write('#Breast Thickness = '+str(self.breast_radius.value)+'\n')
        file.write('#Breast Glandularity = '+str(self.gland.value)+'\n')
        file.write('#Skin Thickness (mm) = '+str(self.skin.value)+'\n')
        file.write('#Adipose Thickness (mm) = '+str(self.adipose.value)+'\n')
        file.write('#CPCD = '+str(self.ccpd.value)+'\n')
        file.write('#Pot.(kV)\tHVL (mmAl)\tDgN\tUncertainty\n')
        for i in range(len(potential)):
            #file.write(' '+str(potential[i])+'\t'+str(np.round(hvl[i],4))+'\t'+str(np.round(dgn[i],5))+'\t'+str(np.round(edgn[i],5))+'\n')
            file.write(f' {potential[i]:2d}\t\t{hvl[i]:6.4f}\t\t{dgn[i]:6.4f}\t{edgn[i]:6.4f}\n')

        file.close()
                   
        
        
                   
        
        

    
    
