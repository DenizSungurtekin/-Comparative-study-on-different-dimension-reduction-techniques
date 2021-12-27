#ifndef HIDIM_KNNGRAPH_H
#define HIDIM_KNNGRAPH_H

#include <hidim/types.h>
#include <hidim/math/spanner/geospanner.h>
#include <hidim/dataset/dataset.h>

namespace hidim
{

class KNNGraph : public GeoSpanner
{
public:
    //KNNGraph();
    //KNNGraph(dim_t K);
    KNNGraph(DataContainer *dataset,dim_t K);
    KNNGraph(DataContainer *dataset,std::ifstream *stream);
    virtual ~KNNGraph();

    void setK(dim_t K);
    dim_t getK();

    void clear();

    void build();
    void build_parallel();

    bool read(ifstream *stream);
    void readHeader(ifstream *stream);

    void write(ofstream *stream);
    void writeHeader(ofstream *stream);

protected:

private:
    dim_t K = 0;
    static void worker(dim_t id, KNNGraph *knn, vertex_id vertex_start, vertex_id vertex_end);
};
}


#endif // HIDIM_KNNGRAPH_H
