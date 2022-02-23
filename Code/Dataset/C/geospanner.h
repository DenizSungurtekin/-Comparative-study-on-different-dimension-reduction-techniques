#ifndef HIDIM_GEOSPANNER_H
#define HIDIM_GEOSPANNER_H

#include <fstream>

#include <hidim/math/graph/graph.h>
#include <hidim/dataset/datacontainer.h>

namespace hidim
{
class GeoSpanner
{
public:
    GeoSpanner();
    GeoSpanner(DataContainer *dataset);
    virtual ~GeoSpanner();

    dim_t M();
    dim_t N();
    dim_t getType();
    void setType(dim_t type);
    std::string getName();
    void setDataset(DataContainer *dataset);
    DataContainer *getDataset();
    void setDatasetID(dim_t ID);
    dim_t getDatasetID();
    void setName(std::string name);
    void setDirected(bool directed);
    void setCheckSimple(bool simple);
    void setHasSelfloop(bool selfloop);

    void addAllNodes(dim_t n);
    bool checkNode(vertex_id id);
    //void addNeighbor(vertex_id from, vertex_id to, edge_weight weight,bool sort = false);
    void addEdge(vertex_id from, vertex_id to, edge_weight weight = 1.,bool sort = false);
    std::vector<vertex_id> getStar(vertex_id i);
    std::vector<node_elt_c> getWStar(vertex_id i);
    std::vector<vertex_id> findLeaf();
    void reverseInto(GeoSpanner *g);
    std::vector<node_c> V();
    node_c V(vertex_id);

    void clear();
    bool check();
    void info();

    bool read(std::ifstream *stream);
    void readHeader(std::ifstream *stream);

    void write(std::ofstream *stream);
    void writeHeader(std::ofstream *stream);


protected:
    Graph graph;
    DataContainer *dataset = NULL;
    dim_t datasetID = 0;
    bool built = false;

    //virtual void build() = 0;

private:

};
}
#endif // HIDIM_GEOSPANNER_H
