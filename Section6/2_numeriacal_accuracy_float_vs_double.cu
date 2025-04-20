#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

template<typename T>
void accuracy_comparison(const char* type_name)
{
    printf("%s accuracy comparison\n", type_name);
    T a = T(3.1415927);
    T b = T(3.1415928);
    
    if (a == b)
    {
        printf("a is equal to b\n");
    }
    else
    {
        printf("a does not equal b\n");
    }
}

int main()
{
    accuracy_comparison<float>("float");
    accuracy_comparison<double>("double");

    return 0;
}