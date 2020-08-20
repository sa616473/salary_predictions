import seaborn as sns
import matplotlib.pyplot as plt


def visualization_plot(feature_x, data, feature_y=None,  hue=None, kde_line=False, size=(14,7), bins=100, plot_type=''):
    '''
    This function plots the specified plot_type if the 
    plot type does not exist it will raise a error.
    plot_type should in the fromat of a string
    '''
    plt.figure(figsize=size)

    if plot_type == 'dist' or plot_type == 'distribution':  
        sns.distplot(data[feature_x],bins=bins, kde=kde_line)
    
    elif plot_type == 'count':
        sns.countplot(x=feature_x, data=data, hue=hue)
    
    elif plot_type == 'violin' and feature_y != None:
        sns.violinplot(x=feature_x, y=feature_y, data=data)
    
    elif plot_type == 'bar':
        sns.barplot(x=feature_x, y=feature_y, data=data)
    
    elif plot_type == 'box' and feature_y !=None:
        sns.boxplot(x=feature_x, y=feature_y, data=data)
    
    else:
        raise ValueError('Invalid plot type\n or\n the type of plot does not exist in the method\n')
        return -1
    if feature_y == None:
        plt.savefig('../reports/figures/graphs/png/{}_{}.png'.format(feature_x, plot_type))
        plt.savefig('../reports/figures/graphs/pdf/{}_{}.pdf'.format(feature_x, plot_type))
    elif feature_y != None:
        plt.savefig('../reports/figures/graphs/png/{}_{}_{}.png'.format(feature_x,feature_y, plot_type))
        plt.savefig('../reports/figures/graphs/pdf/{}_{}_{}.pdf'.format(feature_x,feature_y, plot_type))
