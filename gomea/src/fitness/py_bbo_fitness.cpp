#ifndef CPP_STANDALONE

#include "gomea/src/fitness/py_bbo_fitness.hpp"

namespace gomea{
namespace fitness{

template<>
pyBBOFitnessFunction_t<char>::pyBBOFitnessFunction_t( int number_of_parameters, int alphabet_size, PyObject *obj ) : BBOFitnessFunction_t<char>(number_of_parameters)
{
	this->name = "Your own fitness function (Python)";
	this->py_class = obj;
	this->alphabet_size = alphabet_size;
	this->initialize();
}

template<>
pyBBOFitnessFunction_t<char>::pyBBOFitnessFunction_t( int number_of_parameters, int alphabet_size, double vtr, PyObject *obj ) : BBOFitnessFunction_t<char>(number_of_parameters,vtr)
{
	this->name = "Your own fitness function (Python)";
	this->py_class = obj;
	this->alphabet_size = alphabet_size;
	this->initialize();
}

template<>
pyBBOFitnessFunction_t<double>::pyBBOFitnessFunction_t( int number_of_parameters, PyObject *obj ) : BBOFitnessFunction_t<double>(number_of_parameters)
{
	this->name = "Your own fitness function (Python)";
	this->py_class = obj;
	this->initialize();
}

template<>
pyBBOFitnessFunction_t<double>::pyBBOFitnessFunction_t( int number_of_parameters, double vtr, PyObject *obj ) : BBOFitnessFunction_t<double>(number_of_parameters,vtr)
{
	this->name = "Your own fitness function (Python)";
	this->py_class = obj;
	this->initialize();
}

template<>
double pyBBOFitnessFunction_t<char>::objectiveFunction( int objective_index, vec_t<char> &variables )
{
	double result = gomea_pyfitness_objective_function_bbo_discrete(py_class,objective_index,variables);
	if( result == INFINITY )
		throw std::runtime_error("FitnessFunction does not implement objective_function(int,vector[char]).");
	return result;
}

template<>
double pyBBOFitnessFunction_t<double>::objectiveFunction( int objective_index, vec_t<double> &variables )
{
	double result = gomea_pyfitness_objective_function_bbo_realvalued(py_class,objective_index,variables);
	if( result == INFINITY )
		throw std::runtime_error("FitnessFunction does not implement objective_function(int,vector[double]).");
	return result;
}

template<>
double pyBBOFitnessFunction_t<char>::constraintFunction( vec_t<char> &variables )
{
	double result = gomea_pyfitness_constraint_function_bbo_discrete(py_class,variables);
	if( result == INFINITY )
		throw std::runtime_error("FitnessFunction does not implement constraint_function(vector[char]).");
	return result;
}

template<>
double pyBBOFitnessFunction_t<double>::constraintFunction( vec_t<double> &variables )
{
	double result = gomea_pyfitness_constraint_function_bbo_realvalued(py_class,variables);
	if( result == INFINITY )
		throw std::runtime_error("FitnessFunction does not implement constraint_function(vector[double]).");
	return result;
}

template<class T>
double pyBBOFitnessFunction_t<T>::getLowerRangeBound( int variable_index, vec_t<T> &variables )
{
	int numpy_type_id;
    if (std::is_same<T, double>::value) {
        numpy_type_id = NPY_FLOAT64;
    } else if (std::is_same<T, char>::value) {
        numpy_type_id = NPY_BYTE;
    }

	npy_intp dims[1] = { static_cast<npy_intp>(variables.size()) };
    PyObject *np_array_obj = PyArray_SimpleNewFromData(1, dims, numpy_type_id, variables.data());

	double result = gomea_pyfitness_lower_range_bound(this->py_class, variable_index, np_array_obj);
    
    Py_DECREF(np_array_obj);
    
    return result;
}

template<class T>	
double pyBBOFitnessFunction_t<T>::getUpperRangeBound( int variable_index, vec_t<T> &variables )
{
	int numpy_type_id;
    if (std::is_same<T, double>::value) {
        numpy_type_id = NPY_FLOAT64;
    } else if (std::is_same<T, char>::value) {
        numpy_type_id = NPY_BYTE;
    }

	npy_intp dims[1] = { static_cast<npy_intp>(variables.size()) };
    PyObject *np_array_obj = PyArray_SimpleNewFromData(1, dims, numpy_type_id, variables.data());

	double result = gomea_pyfitness_upper_range_bound(this->py_class, variable_index, np_array_obj);
    
    Py_DECREF(np_array_obj);
    
    return result;
}

template<>
double pyBBOFitnessFunction_t<char>::getLowerRangeBound( int variable_index, vec_t<char> &variables )
{
	npy_intp dims[1] = { static_cast<npy_intp>(variables.size()) };
    PyObject *np_array_obj = PyArray_SimpleNewFromData(1, dims, NPY_BYTE, variables.data());

	double result = gomea_pyfitness_lower_range_bound(this->py_class, variable_index, np_array_obj);
    
    Py_DECREF(np_array_obj);
    
    if( result == -INFINITY ){
        return 0;
    }
    return result;
}

template<>	
double pyBBOFitnessFunction_t<char>::getUpperRangeBound( int variable_index, vec_t<char> &variables )
{
	npy_intp dims[1] = { static_cast<npy_intp>(variables.size()) };
    PyObject *np_array_obj = PyArray_SimpleNewFromData(1, dims, NPY_BYTE, variables.data());

	double result = gomea_pyfitness_upper_range_bound(this->py_class, variable_index, np_array_obj);
    
    Py_DECREF(np_array_obj);
    
    if( result == INFINITY ){
        return this->alphabet_size-1;
    }
    return result;
}

template class pyBBOFitnessFunction_t<char>;
template class pyBBOFitnessFunction_t<double>;

}}

#endif