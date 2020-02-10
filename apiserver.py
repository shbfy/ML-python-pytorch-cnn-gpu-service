# Import your handlers here
from service import MNIST, Intro


# Configuration for web API implementation
def config(api):

    # Instantiate handlers and map routes
    api.add_route('/mnist', Intro())
    api.add_route('/mnist/{index:int(min=0)}', MNIST())
