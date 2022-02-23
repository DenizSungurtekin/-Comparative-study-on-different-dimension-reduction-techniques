#ifndef HIDIM_DATA_H
#define HIDIM_DATA_H

#include <Eigen/Dense>

#include <hidim/types.h>

namespace hidim
{

class Data : public Eigen::VectorXd
{
public:
    Data();
    virtual ~Data();
    Data(const Data& other);
    Data(Eigen::VectorXd vector);

protected:
private:
};

}

#endif // HIDIM_DATA_H
