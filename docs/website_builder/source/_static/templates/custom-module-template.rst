{{ name | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :special-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Variables') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

