import matplotlib.pyplot as plt

def memplot(df,draw_type=None):
    if draw_type :
        df=df[df['hook_type']==draw_type]
    pl=df.plot(y='mem_all')
    pl.get_figure().savefig('./mem_all.png')
    return