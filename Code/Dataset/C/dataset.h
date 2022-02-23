#ifndef HIDIM_DATASET_H
#define HIDIM_DATASET_H

#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

#include <hidim/types.h>
#include <hidim/dataset/datacontainer.h>
#include <hidim/dataset/data.h>
#include <hidim/dataset/datatype.h>
#include <hidim/dataset/transform.h>

namespace hidim
{

class Dataset : public DataContainer, public Eigen::MatrixXd
{
public:
    Dataset();
    Dataset(dim_t lN,dim_t lD);
    virtual ~Dataset();

//    void setName(std::string theName);
//    std::string getName();

//    void setN(dim_t theN);
//    dim_t getN();
//    void setD(dim_t theD);
//    dim_t getD();

    dim_t N();
    dim_t D();
    hidim::Data data(dim_t i);
    bool setValue(dim_t n,dim_t d,double value);
    double getValue(dim_t n,dim_t d);
    //dim_t getID();

    void adjustSize();
    void setSize(dim_t N,dim_t D);
    Transform *getTransform();
    void setHeader(std::vector<std::string> lHeader);
    std::vector<std::string> getHeader();

    void clear();

//    void addSpanner(GeoSpanner *g);


    // TODO: get mean, variance per feature
    // TODO: get mean, variance of interdistance -> use knn ??


    void setRandomTranslation(double mini, double maxi);
    void setUniformTranslation(double value);
    void setTranslation(Eigen::VectorXd theTranslation);

    void setRandomScaling(double mini, double maxi);
    void setUniformScaling(double value);
    void setScaling(Eigen::VectorXd theScaling);

    void setRotation(bool theRotation);

    void open(std::string theName,dim_t N, dim_t D);
    void close();

    void addCode(std::vector<std::string> command);
    void addPart(Datatype *theDatatype);
    bool checkParts();
    void generation(dim_t N, dim_t D);

    Dataset *read(std::ifstream *inStream);
    void write(std::ofstream *outStream);
    void print();

protected:


private:
    bool isOpen = false;
    std::string code;
    std::vector<std::string> header;
    std::vector<Datatype *> part;
    Transform transform;

};

}

#endif // HIDIM_DATASET_H
