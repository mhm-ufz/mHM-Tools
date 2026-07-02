{% set shortname = fullname.split('.')[-1] %}
{% if fullname in ["mhm_tools.common", "mhm_tools.post", "mhm_tools.pre"] %}
{% set title = shortname | capitalize %}
{% else %}
{% set title = shortname %}
{% endif %}
{{ title | escape | underline}}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

{% if classes %}
Classes
-------

.. autosummary::
   :toctree:

{% for item in classes %}
   ~{{ fullname }}.{{ item }}
{% endfor %}
{% endif %}

{% if functions %}
Functions
---------

.. autosummary::
   :toctree:

{% for item in functions %}
   ~{{ fullname }}.{{ item }}
{% endfor %}
{% endif %}

.. raw:: latex

    \clearpage
