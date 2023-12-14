:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ fullname | replace("sdr.", "sdr::") }}

{# In the fullname (e.g. `sdr.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `sdr::methodname`)
specifies `sdr` as the module name. #}