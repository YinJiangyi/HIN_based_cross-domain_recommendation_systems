# Heterogeneous_information_network_based_recommendation_systems
we designed a personalized rating prediction system(HecRec) and a top-k list recommendation model (EPCDRec) which explore both heterogeneous information and cross-domain recommendation.

## HecRec
We propose a HIN Embedding based Cross-domain Recommendation (HecRec) framework, which exploits cross-domain information by establishing meta-path based HIN embeddings in both the source and the target domain and predicts personalized ratings by integrating the obtained HIN embeddings with a rating predictor. To make the best use of cross-domain information and avoid the knowledge confliction between knowledge from different meta- paths observed in real-system datasets, we adopt a concept of “overpass bridge” to integrate the HIN embeddings drawn via different meta-paths.

## EPCDRec
For the top-k list recommendation senario, we designed a model based network embedding propagation layers(npl). We use the neural network which consists of multiple npls to modelding the message flow on the crosss-domain HIN, we conduct training of the network repersentation of users and item, and explore the presonalized feature at the same time in the end-to-end model.
