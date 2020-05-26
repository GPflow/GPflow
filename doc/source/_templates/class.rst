{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   {#:private-members:#}
   {#:special-members:#}
   {#:undoc-members:#}

   {% block methods %}
   {% if methods %}
   .. rubric:: Class methods

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Class attributes

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
