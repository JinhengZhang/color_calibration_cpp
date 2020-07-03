#include <map>
#include <iostream>
#include <string>
#include "colorspace.h"

using namespace std;

typedef void* (*PTRCreateObject)(void);

class ClassFactory {
private:
    map<String, PTRCreateObject> m_classMap;
    ClassFactory() {}; 

public:
    void* getClassByName(String className);
    void registClass(String name, PTRCreateObject method);
    static ClassFactory& getInstance();
};

void* ClassFactory::getClassByName(String className) {
    map<String, PTRCreateObject>::const_iterator iter;
    iter = m_classMap.find(className);
    if (iter == m_classMap.end())
        return NULL;
    else
        return iter->second();
}

void ClassFactory::registClass(String name, PTRCreateObject method) {
    m_classMap.insert(pair<String, PTRCreateObject>(name, method));
}

ClassFactory& ClassFactory::getInstance() {
    static ClassFactory sLo_factory;
    return sLo_factory;
}

class RegisterAction {
public:
    RegisterAction(String className, PTRCreateObject ptrCreateFn) {
        ClassFactory::getInstance().registClass(className, ptrCreateFn);
    }
};

#define REGISTER(className)                                             \
    className* objectCreator##className(){return new className; }      \
    RegisterAction g_creatorRegister##className(#className, (PTRCreateObject)objectCreator##className)

REGISTER(sRGB_Base)
