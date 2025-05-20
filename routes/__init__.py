from .index import register_routes as register_index_routes
from .association_rules import register_routes as register_association_routes
from .decision_tree import register_routes as register_decision_tree_routes
from .reduct import register_routes as register_reduct_routes
from .kmeans import register_routes as register_clustering_routes
from .naive_bayes import register_routes as register_naive_bayes_routes
from .naive_bayes import register_routes as register_naive_bayes_routes
from .kohonen import register_routes as register_kohonen_routes

def register_all_routes(app):
    register_index_routes(app)
    register_association_routes(app)
    register_decision_tree_routes(app)
    register_reduct_routes(app)
    register_clustering_routes(app)
    register_kohonen_routes(app)
    register_naive_bayes_routes(app)