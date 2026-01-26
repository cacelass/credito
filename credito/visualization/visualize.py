import matplotlib.pyplot as plt
import seaborn as sns
from credito.utils.paths import FIGURES_DIR

def plot_age_distribution(df):
    """Genera y guarda la gráfica de distribución de edad."""
    print("--> Generando visualizaciones...")
    
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    
    # Boxplot
    sns.boxplot(x='age', data=df, orient='v', ax=ax1)
    ax1.set_xlabel('People Age', fontsize=15)
    ax1.set_ylabel('Age', fontsize=15)
    ax1.set_title('Age Distribution', fontsize=15)
    ax1.tick_params(labelsize=15)

    # Distplot
    sns.distplot(df['age'], ax=ax2)
    sns.despine(ax=ax2)
    ax2.set_xlabel('Age', fontsize=15)
    ax2.set_ylabel('Occurence', fontsize=15)
    ax2.set_title('Age x Occurence', fontsize=15)
    ax2.tick_params(labelsize=15)

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "age_distribution.png"
    plt.savefig(output_path)
    print(f"    Gráfica guardada en {output_path}")
    plt.close()