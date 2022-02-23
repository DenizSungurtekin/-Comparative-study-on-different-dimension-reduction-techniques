#ifndef HIDIM_DATACONTAINER_H
#define HIDIM_DATACONTAINER_H

#define HIDIM_DATACONTAINER_NOTYPE 0
#define HIDIM_DATACONTAINER_DATASET 1
#define HIDIM_DATACONTAINER_SUBSET 2

#include <vector>

#include <hidim/types.h>
#include <hidim/dataset/data.h>

namespace hidim
{

class GeoSpanner;

class DataContainer
{
public:
    DataContainer();
    virtual ~DataContainer();

    void setType(dim_t type);
    virtual dim_t N() = 0;
    virtual dim_t D() = 0;
    virtual hidim::Data data(dim_t i) =0 ;
    virtual bool setValue(dim_t n,dim_t d,double value) = 0;
    virtual double getValue(dim_t n,dim_t d) = 0;

    //virtual void adjustSize();  // TODO: set as pure virtual functions
    //virtual void setSize(dim_t N,dim_t D);  // TODO: set as pure virtual functions

    dim_t getID();
    void setName(std::string theName);
    std::string getName();

    virtual void clear(); // TODO: set as pure virtual functions

    void addSpanner(GeoSpanner *g);

protected:
    dim_t id = 0;
    //dim_t lD = 0;
    std::string name;
    std::vector<GeoSpanner *> spanner;

private:
    dim_t type = HIDIM_DATACONTAINER_NOTYPE;
};

}

#endif // HIDIM_DATACONTAINER_H
