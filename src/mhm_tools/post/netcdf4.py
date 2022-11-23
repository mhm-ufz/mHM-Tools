"""
Author
------
David Schaefer

Purpose
-------
A sanitizing layer for the netCDF4 library. Adds a number of convenince methods
and aims for a cleaner user interface. All classes avaliable are children of their
netCDF4 counterparts.
"""

import uuid
from collections import OrderedDict

from netCDF4 import (
    Dataset,
    Dimension,
    Group,
    Variable,
    chartostring,
    date2index,
    date2num,
    getlibversion,
    num2date,
    stringtoarr,
    stringtochar,
)


def _tupelize(arg):
    """

    Parameters
    ----------
    arg :


    Returns
    -------

    """
    if isinstance(arg, str):
        return (arg,)
    try:
        return tuple(arg)
    except TypeError:
        return (arg,)


def copyGroup(
    ncin,
    group,
    skipdims=None,
    skipgroups=None,
    skipvars=None,
    skipattrs=None,
    fixdims=False,
    vardata=False,
    varparams=None,
):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createGroup method
    group :
        Instance of an object with dimensions
    skipdims :
        optional (Default value = None)
    skipgroups :
        optional (Default value = None)
    skipvars :
        optional (Default value = None)
    skipattrs :
        optional (Default value = None)
    fixdims :
        optional (Default value = False)
    vardata :
        optional (Default value = False)
    varparams :
        optional (Default value = None)

    Returns
    -------
    type
        ------
        NcGroup

        Purpose
        -------
        Copy the given group to ncin

    """
    out = ncin.createGroup(group.name)
    out.set_fill_off()
    out.copyDimensions(group.dimensions, skip=skipdims, fix=fixdims)
    out.copyVariables(group.variables, skip=skipvars, data=vardata, varparams=varparams)
    out.copyAttributes(group.attributes, skipattrs)
    out.copyGroups(group.groups, skipgroups)
    return out


def copyDataset(
    ncin,
    group,
    skipdims=None,
    skipgroups=None,
    skipvars=None,
    skipattrs=None,
    fixdims=False,
    vardata=False,
    varparams=None,
):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createGroup method
    group :
        Instance of an object with dimensions
    skipdims :
        optional (Default value = None)
    skipgroups :
        optional (Default value = None)
    skipvars :
        optinal (Default value = None)
    skipattrs :
        optinal (Default value = None)
    fixdims :
        optional (Default value = False)
    vardata :
        optional (Default value = False)
    varparams :
        optional (Default value = None)

    Returns
    -------
    type
        ------
        NcDataset/NcGroup

        Purpose
        -------
        Copy the content of given group to ncin

    """
    ncin.set_fill_off()
    ncin.copyDimensions(group.dimensions, skip=skipdims, fix=fixdims)
    ncin.copyVariables(
        group.variables, skip=skipvars, data=vardata, varparams=varparams
    )
    ncin.copyAttributes(group.attributes, skipattrs)
    ncin.copyGroups(group.groups, skipgroups)
    return ncin


def copyGroups(ncin, groups, skip=None):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createGroup method
    groups :
        Dictionary
    skip :
        optional (Default value = None)

    Returns
    -------
    type
        ------
        None

        Purpose
        -------
        Copy the given groups to ncin

    """
    for g in groups.values():
        if g.name not in _tupelize(skip):
            ncin.copyGroup(g)


def copyDimension(ncin, dim, fix=False, fail=True):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createDimension method
    dim :

    fix :
        optional (Default value = False)
    fail :
        Optional (Default value = True)

    Returns
    -------
    type
        ------
        netCDF4.Dimension

        Purpose
        -------
        Copy the given dimension to ncin

    """
    length = None if dim.isunlimited() and not fix else len(dim)
    try:
        return ncin.createDimension(dim.name, length)
    except Exception:
        if fail:
            raise
        return ncin.dimensions[dim.name]


def copyDimensions(ncin, dimensions, skip=None, fix=False, fail=True):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createDimension method
    dimensions :

    skip :
        optional (Default value = None)
    fix :
        optional (Default value = False)
    fail :
        Optional (Default value = True)

    Returns
    -------
    type
        ------
        None

        Purpose
        -------
        Copy the given dimensions to ncin

    """
    for d in dimensions.values():
        if d.name not in _tupelize(skip):
            ncin.copyDimension(d, fix=fix, fail=fail)


def copyAttributes(ncin, attributes, skip=None):
    """Arguments
    ---------
    ncin              : Instance of an object with a createAttribute method
                        (i.e. NcDataset, NcGroup, NcVariable)
    attributes        : Dictionary
                        key   : string
                        value : string/any numeric type
    skip (optional)   : string or list/tuple of strings
                        Name(s) of attribute(s) to skip

    Return
    ------
    None

    Purpose
    -------
    Copy the given attributes to ncin

    Parameters
    ----------
    ncin :

    attributes :

    skip :
         (Default value = None)

    Returns
    -------


    """
    for k, v in attributes.items():
        if k not in _tupelize(skip):
            if k == "missing_value":
                try:
                    v = ncin.dtype.type(v)
                except Exception:
                    pass
            ncin.createAttribute(k, v)


def copyVariable(ncin, var, data=True, dims=False, fail=True, **kwargs):
    """Arguments
    ---------
    ncin            : Instance of an object with a createCopy method
                      (i.e. NcDataset, NcGroup, NcVariable)
    var             : Instance of NcVariable
    data (optional) : boolean, copy variable data
    dims (Optional[bool]): copy missing dimensions
    fail (Optional[bool]): raise an exception if variable exists, ignored at the moment
    kwargs          : will be passed to createVariable. Allows to set
                      parameters like chunksizes, deflate_level, ...

    Return
    ------
    NcVariable

    Purpose
    -------
    Copy the given variables to ncin. Copy the data if data=True

    Parameters
    ----------
    ncin :

    var :

    data :
         (Default value = True)
    dims :
         (Default value = False)
    fail :
         (Default value = True)
    **kwargs :


    Returns
    -------


    """
    invardef = var.definition

    if data is not True:
        invardef["chunksizes"] = None
    invardef.update(kwargs)

    if dims:
        nc = var.parent
        try:
            shape = data.shape
        except AttributeError:
            shape = var.shape
        for name, length in zip(var.dimensions, shape):
            l = None if nc.dimensions[name].isunlimited() else length
            ncin.createDimension(name, l, fail=fail)

    vname = invardef.pop("name")
    try:
        invar = ncin.createVariable(vname, invardef.pop("dtype"), **invardef)
    except RuntimeError:
        if fail:
            raise
        invar = ncin.variables[vname]

    invar.copyAttributes(var.attributes)
    if data is True and var.shape:
        invar[:] = var[:]
    elif data is not False:
        # i.e. if an array is given
        invar[:] = data
    return invar


def copyVariables(
    ncin, variables, skip=None, data=True, dims=False, fail=True, varparams=None
):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a createCopy method
    variables :
        Dictionary
    skip :
        optional (Default value = None)
    data :
        optional (Default value = True)
    dims :
        Optional (Default value = False)
    fail :
        Optional (Default value = True)
    varparams :
        optional (Default value = None)

    Returns
    -------
    type
        ------
        NcVariable

        Purpose
        -------
        Copy the given variables to ncin. Copy the data if data=True

    """

    if varparams is None:
        varparams = dict()
    for v in variables.values():
        if v.name not in _tupelize(skip):
            ncin.copyVariable(v, data, dims, fail, **varparams)


def createDimensions(ncin, dim_dict, fail=True):
    """

    Parameters
    ----------
    ncin :

    dim_dict :

    fail :
         (Default value = True)

    Returns
    -------

    """
    for name, length in dim_dict.items():
        ncin.createDimension(name, length, fail=fail)


def getVariableDefinition(ncvar):
    """

    Parameters
    ----------
    ncvar :


    Returns
    -------

    """
    out = ncvar.filters() if ncvar.filters() else {}
    out.update(
        {
            "name": ncvar.name,
            "dtype": ncvar.dtype,
            "dimensions": ncvar.dimensions,
            "chunksizes": ncvar.chunking()
            if not isinstance(ncvar.chunking(), str)
            else None,
            "fill_value": getattr(ncvar, "_FillValue", None),
        }
    )
    return out


def getDates(ncin, timesteps=None, timevar="time", units=None, calendar=None):
    """

    Parameters
    ----------
    ncin :
        Instance of an object holding variables
    timesteps :
        optional (Default value = None)
    timevar :
         (Default value = "time")
    units :
        optional (Default value = None)
    calendar :
        name following the CF conventions (Default value = None)

    Returns
    -------
    type


    """
    var = ncin.variables[timevar]
    if not units:
        try:
            units = var.units
        except AttributeError:
            raise AttributeError(
                "Time variable does not specify an units attribute! Pass as argument."
            )

    if not calendar:
        try:
            calendar = var.calendar
        except AttributeError:
            calendar = "standard"

    if not timesteps:
        timesteps = var[:]

    dates = num2date(timesteps, units, calendar)

    try:
        return [d.date() for d in dates]
    except AttributeError:
        return dates


def setFillValue(ncin, value):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a _FillValue attribute
    value :


    Returns
    -------
    type


    """
    ncin.setncattr("_FillValue", value)


def getFillValue(ncin):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a _FillValue attribute

    Returns
    -------
    type


    """
    try:
        return ncin.getncattr("_FillValue")
    except AttributeError:
        return None


def setAttribute(ncin, name, value):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a setncatts method
    name :
        string
    value :
        string or any numeric type

    Returns
    -------
    type
        ------
        None

        Purpose
        -------
        Set/Write the attribute given as name, value

    """
    ncin.setncattr(name, value)


def setAttributes(ncin, attdict):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a setncatts method
    attdict :
        dictionary

    Returns
    -------
    type
        ------
        None

        Purpose
        -------
        Set/Write the attributes given in attdict

    """
    ncin.setncatts(attdict)


def filterVariables(ncin, dims=None, ndim=None):
    """

    Parameters
    ----------
    ncin :
        Instance of an object with a variables attribute
    dims :
        optional (Default value = None)
    ndim :
        optional (Default value = None)

    Returns
    -------
    type
        and/or have ndims dimensions.

    """
    out = OrderedDict()
    dims = set(dims or {})

    for v in ncin.variables.values():
        if dims.issubset(set(v.dimensions)):
            if ndim:
                if ndim == len(v.dimensions):
                    out[v.name] = v
            else:
                out[v.name] = v

    return out


def filterDimensions(ncin, lengths):
    """

    Parameters
    ----------
    ncin :

    lengths :
        integer

    Returns
    -------
    type


    """
    try:
        lengths[0]
    except TypeError:
        lengths = (lengths,)

    return OrderedDict(
        [(d.name, d) for d in ncin.dimensions.values() if len(d) in lengths]
    )


def getGroups(ncin):
    """

    Parameters
    ----------
    ncin :


    Returns
    -------

    """
    out = OrderedDict()
    for g in getattr(ncin, "groups").values():
        out[g.name] = NcGroup(ncin, g.name, id=g._grpid)
    return out


def getVariables(ncin):
    """

    Parameters
    ----------
    ncin :


    Returns
    -------

    """
    out = OrderedDict()
    for v in getattr(ncin, "variables").values():
        out[v.name] = NcVariable(ncin, v.name, v.dtype, v.dimensions, id=v._varid)
    return out


def getDimensions(ncin):
    """

    Parameters
    ----------
    ncin :


    Returns
    -------

    """

    out = OrderedDict()
    for dim in getattr(ncin, "dimensions").values():
        out[dim.name] = NcDimension(ncin, dim.name, id=dim._dimid)
    return out


def getAttributes(ncin):
    """

    Parameters
    ----------
    ncin :


    Returns
    -------

    """
    out = OrderedDict()
    for k in ncin.ncattrs():
        if not k.startswith("_"):
            out[k] = ncin.getncattr(k)
    return out


def getParent(ncin):
    """

    Parameters
    ----------
    ncin :


    Returns
    -------

    """
    return ncin._grp


def attributeSetter(ncin, name, value):
    """

    Parameters
    ----------
    ncin :

    name :

    value :


    Returns
    -------

    """
    ncin.__dict__[name] = value


def attributeGetter(ncin, name):
    """

    Parameters
    ----------
    ncin :

    name :


    Returns
    -------

    """
    try:
        return ncin.__dict__[name]
    except KeyError:
        try:
            return getattr(super(ncin.__class__, ncin), name)
        except KeyError:
            raise AttributeError(
                "'{:}' object has no attribute '{:}'".format(ncin.__class__, name)
            )


def createGroup(ncin, name):
    """

    Parameters
    ----------
    ncin :

    name :


    Returns
    -------

    """
    grp = NcGroup(ncin, name)
    ncin.groups[name] = grp
    return grp


def createVariable(ncin, *args, **kwargs):
    """

    Parameters
    ----------
    ncin :

    *args :

    **kwargs :


    Returns
    -------

    """
    var = NcVariable(ncin, *args, **kwargs)
    ncin.variables[var.name] = var
    return var


def createDimension(ncin, name, length, fail=True):
    """

    Parameters
    ----------
    ncin :

    name :

    length :

    fail :
         (Default value = True)

    Returns
    -------

    """
    try:
        dim = NcDimension(ncin, name, length)
        ncin.dimensions[dim.name] = dim
    except Exception:
        if fail:
            raise
        dim = ncin.dimensions[name]
    return dim


class NcDataset(Dataset):
    """ """

    def __init__(
        self,
        filename=None,
        mode="r",
        clobber=True,
        diskless=False,
        persist=False,
        weakref=False,
        format="NETCDF4",
    ):

        if filename is None:
            # in memory dataset
            filename = str(uuid.uuid4())
            mode = "w"
            diskless = True

        super(NcDataset, self).__init__(
            filename=filename,
            mode=mode,
            clobber=clobber,
            diskless=diskless,
            persist=persist,
            weakref=weakref,
            format=format,
        )

        self.fname = filename
        for k, v in zip(self.groups, getGroups(self).values()):
            self.groups[k] = v
        for k, v in zip(self.dimensions, getDimensions(self).values()):
            self.dimensions[k] = v
        for k, v in zip(self.variables, getVariables(self).values()):
            self.variables[k] = v

    def tofile(self, fname):
        """

        Parameters
        ----------
        fname :


        Returns
        -------

        """
        # preserve dataset options
        with NcDataset(fname, "w") as out:
            out.copyDataset(self, vardata=True)

    # def __enter__(self):
    #     return self

    # def __exit__(self, *args, **kwargs):
    #     self.close()

    copyDataset = copyDataset
    copyDimension = copyDimension
    copyDimensions = copyDimensions
    copyAttributes = copyAttributes
    copyVariable = copyVariable
    copyVariables = copyVariables
    copyGroup = copyGroup
    copyGroups = copyGroups
    createAttribute = setAttribute
    createAttributes = setAttributes
    createDimensions = createDimensions
    createVariable = createVariable
    createGroup = createGroup
    createDimension = createDimension
    filterVariables = filterVariables
    filterDimensions = filterDimensions
    getDates = getDates
    attributes = property(fget=getAttributes)
    # restore a "normal" attribute access behaviour
    __setattr__ = attributeSetter
    __getattr__ = attributeGetter


class NcGroup(Group):
    """ """

    def __init__(self, *args, **kwargs):
        super(NcGroup, self).__init__(*args, **kwargs)
        for k, v in zip(self.groups, getGroups(self).values()):
            self.groups[k] = v
        for k, v in zip(self.dimensions, getDimensions(self).values()):
            self.dimensions[k] = v
        for k, v in zip(self.variables, getVariables(self).values()):
            self.variables[k] = v

    copyDimension = copyDimension
    copyDimensions = copyDimensions
    copyAttributes = copyAttributes
    copyVariable = copyVariable
    copyVariables = copyVariables
    copyGroup = copyGroup
    copyGroups = copyGroups
    createAttribute = setAttribute
    createAttributes = setAttributes
    createDimensions = createDimensions
    createVariable = createVariable
    createGroup = createGroup
    createDimension = createDimension
    filterVariables = filterVariables
    filterDimensions = filterDimensions
    getDates = getDates
    attributes = property(fget=getAttributes)
    parent = property(fget=getParent)
    # restore a "normal" attribute access behaviour
    __setattr__ = attributeSetter
    __getattr__ = attributeGetter


class NcVariable(Variable):
    """ """

    def __init__(self, *args, **kwargs):
        super(NcVariable, self).__init__(*args, **kwargs)

    copyAttributes = copyAttributes
    createAttribute = setAttribute
    createAttributes = setAttributes
    attributes = property(fget=getAttributes)
    definition = property(fget=getVariableDefinition)
    fill_value = property(fget=getFillValue, fset=setFillValue)
    parent = property(fget=getParent)
    # restore a "normal" attribute access behaviour
    __setattr__ = attributeSetter
    __getattr__ = attributeGetter


# Just to be consistent...
class NcDimension(Dimension):
    """ """

    def __init__(self, *args, **kwargs):
        super(NcDimension, self).__init__(*args, **kwargs)

    parent = property(fget=getParent)
