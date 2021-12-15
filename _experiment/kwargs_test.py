def new(quant=False):
    if quant:
        print("new")

def new2(h=1, **kwargs):
    new(**kwargs)

new2(quant=False)