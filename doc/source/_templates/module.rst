{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree: .
      :nosignatures:
      :template: class.rst
   {% for item in classes %}
      {%- if not item.startswith('_') %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree: .
      :nosignatures:
   {% for item in functions %}
      {#- if not item.startswith('_') #}
      {{ item }}
      {#- endif -#}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree: .
      :nosignatures:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. autopackagesummary:: {{ fullname }}
      :toctree: .
      :template: module.rst
