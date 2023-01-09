{{ name | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


