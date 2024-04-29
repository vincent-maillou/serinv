:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

attribute

.. auto{{ objtype }}:: {{ fullname | replace("serinv.", "serinv::") }}

{# In the fullname (e.g. `serinv.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `serinv::methodname`)
specifies `serinv` as the module name. #}