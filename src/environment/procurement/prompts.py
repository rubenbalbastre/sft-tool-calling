"""Multilingual surface realizations for structured procurement scenarios."""

from copy import deepcopy


LANGUAGES = ("English", "Spanish", "German", "French")
TEMPLATES_PER_LANGUAGE = 25


ROUTES = {
    "English": {
        "direct_supplier": "Evaluate only supplier {requested_supplier_id}.",
        "open_search": "Search for the best supplier option.",
        "compliance_first": "Use only suppliers certified to ISO-14001.",
        "preferred_with_fallback": "Request an offer from {preferred_supplier_id} first, then use another supplier if it cannot satisfy every requirement.",
        "no_feasible_option": "If all available options violate the requirements, report that no feasible option exists.",
    },
    "Spanish": {
        "direct_supplier": "Evalúa únicamente al proveedor {requested_supplier_id}.",
        "open_search": "Busca la mejor opción entre los proveedores disponibles.",
        "compliance_first": "Utiliza solo proveedores con certificación ISO-14001.",
        "preferred_with_fallback": "Solicita primero una oferta a {preferred_supplier_id} y recurre a otro proveedor si no cumple todos los requisitos.",
        "no_feasible_option": "Si todas las opciones disponibles incumplen los requisitos, indica que no existe ninguna opción viable.",
    },
    "German": {
        "direct_supplier": "Prüfe ausschließlich den Lieferanten {requested_supplier_id}.",
        "open_search": "Suche die beste Option unter den verfügbaren Lieferanten.",
        "compliance_first": "Berücksichtige nur Lieferanten mit ISO-14001-Zertifizierung.",
        "preferred_with_fallback": "Fordere zuerst ein Angebot von {preferred_supplier_id} an und weiche auf einen anderen Lieferanten aus, falls nicht alle Anforderungen erfüllt werden.",
        "no_feasible_option": "Falls alle verfügbaren Optionen die Anforderungen verletzen, melde, dass keine geeignete Option existiert.",
    },
    "French": {
        "direct_supplier": "Évalue uniquement le fournisseur {requested_supplier_id}.",
        "open_search": "Recherche la meilleure option parmi les fournisseurs disponibles.",
        "compliance_first": "Utilise uniquement des fournisseurs certifiés ISO-14001.",
        "preferred_with_fallback": "Demande d'abord une offre à {preferred_supplier_id}, puis choisis un autre fournisseur s'il ne respecte pas toutes les contraintes.",
        "no_feasible_option": "Si toutes les options disponibles enfreignent les contraintes, indique qu'aucune option viable n'existe.",
    },
}


PROMPT_TEMPLATES = {
    "English": (
        "We need {quantity} {unit} of {material_id} delivered to {destination} by {required_date}. Total cost must not exceed EUR {budget}, and delivery reliability must be at least {reliability}. {route} Prefer lower carbon emissions, then lower cost. Use the tools and finish with a submitted plan or a no-feasible-option report.",
        "Please arrange {quantity} {unit} of {material_id} for {destination}, arriving no later than {required_date}. Keep the complete order below EUR {budget} with delivery reliability of {reliability} or higher. {route} Of the feasible choices, prioritize carbon impact before price. Complete the task through the tools.",
        "Can you source {material_id} for us? The requirement is {quantity} {unit} in {destination} by {required_date}, with a maximum total spend of EUR {budget} and at least {reliability} delivery reliability. {route} Minimize emissions first and cost second, then submit the result using the tools.",
        "Procurement request: {material_id}, {quantity} {unit}, destination {destination}, deadline {required_date}. Hard limits are EUR {budget} total and {reliability} minimum delivery reliability. {route} Among valid options, favor lower carbon and then lower cost. Resolve the request with the available tools.",
        "For delivery to {destination} by {required_date}, obtain {quantity} {unit} of {material_id}. Do not exceed EUR {budget}; require delivery reliability of at least {reliability}. {route} Carbon reduction is the primary preference and price is secondary. Finish with the appropriate terminal tool.",
        "I need help placing an order for {quantity} {unit} of {material_id}. It has to reach {destination} on or before {required_date}, cost no more than EUR {budget} in total, and achieve at least {reliability} delivery reliability. {route} Prefer the greener feasible choice, using price as the next criterion. Handle the request with the available tools.",
        "Find a compliant way to supply {destination} with {quantity} {unit} of {material_id} by {required_date}. The all-in budget is EUR {budget} and the delivery reliability floor is {reliability}. {route} Once the mandatory conditions are met, optimize for carbon and then cost. Return the final decision through a terminal tool.",
        "We are planning a shipment of {material_id}: {quantity} {unit} are required in {destination} by {required_date}. Reject anything above EUR {budget} total or below {reliability} delivery reliability. {route} For the remaining choices, lower emissions matter more than lower price. Use the tools to complete the selection.",
        "Source {quantity} {unit} of {material_id} for {destination}. Deadline: {required_date}; total budget ceiling: EUR {budget}; minimum delivery reliability: {reliability}. {route} Choose by lowest carbon first and lowest cost second, subject to those limits, and submit the outcome with the tools.",
        "Could you compare the available procurement options for {quantity} {unit} of {material_id}, needed at {destination} by {required_date}? We cannot spend over EUR {budget} overall or accept delivery reliability below {reliability}. {route} Sustainability is the main preference, followed by cost. Finish the workflow using the tools.",
        "Arrange a procurement plan for {quantity} {unit} of {material_id} to {destination} by {required_date}. The total must stay within EUR {budget}, with delivery reliability no lower than {reliability}. {route} Rank feasible results by carbon footprint and then cost. Complete the process using the tools.",
        "Our site in {destination} requires {quantity} {unit} of {material_id} no later than {required_date}. Apply a total spending cap of EUR {budget} and a minimum delivery reliability of {reliability}. {route} Prefer the lowest-emission valid option, breaking ties by cost. Submit the final outcome through the tools.",
        "Identify a valid source for {quantity} {unit} of {material_id}, delivered to {destination} by {required_date}. Enforce EUR {budget} as the maximum all-in cost and {reliability} as the reliability floor. {route} Optimize carbon before price and close the request with a terminal tool.",
        "Please handle this purchasing need: {quantity} {unit} of {material_id} for {destination} by {required_date}. The order must cost at most EUR {budget} and meet at least {reliability} delivery reliability. {route} Of the compliant options, choose for lower emissions first and lower cost second. Use the tools to finish.",
        "Select a supplier and shipment for {material_id}: {quantity} {unit} must arrive in {destination} by {required_date}. Total expenditure is limited to EUR {budget}, and reliability cannot fall below {reliability}. {route} Use carbon impact as the primary preference and cost as the secondary one. Finalize with the tools.",
        "We have an urgent sourcing requirement for {quantity} {unit} of {material_id} in {destination} by {required_date}. Stay under the EUR {budget} total budget and require at least {reliability} delivery reliability. {route} Among feasible plans, minimize emissions before cost. Record the decision through the tools.",
        "Review procurement possibilities for {material_id}, quantity {quantity} {unit}, shipped to {destination} by {required_date}. Exclude plans above EUR {budget} total or below {reliability} reliability. {route} Then favor greener delivery and, secondarily, lower cost. Finish through the available tools.",
        "Build the best compliant purchasing option for {quantity} {unit} of {material_id} going to {destination}. It must arrive by {required_date}, remain within EUR {budget}, and provide at least {reliability} delivery reliability. {route} Prefer carbon savings over price savings. Complete the tool workflow.",
        "Determine how to obtain {quantity} {unit} of {material_id} for {destination} before {required_date}. Treat EUR {budget} and {reliability} delivery reliability as hard constraints. {route} Compare feasible choices by emissions first, then total cost, and submit the result with the tools.",
        "Purchase request for {destination}: supply {quantity} {unit} of {material_id} by {required_date}. No option may exceed EUR {budget} in total or provide less than {reliability} reliability. {route} Choose the most sustainable eligible plan, using cost next. End with the correct terminal tool.",
        "Find and submit a workable plan for delivering {quantity} {unit} of {material_id} to {destination} no later than {required_date}. The ceiling is EUR {budget} total and the reliability threshold is {reliability}. {route} Favor lower carbon, followed by lower cost. Perform the full task with tools.",
        "Assess how we should source {material_id} for {destination}: we need {quantity} {unit} by {required_date}. Keep total cost at or below EUR {budget} and delivery reliability at or above {reliability}. {route} Sustainability outranks price among valid alternatives. Resolve and finalize using tools.",
        "Complete this order using the available procurement tools: {quantity} {unit} of {material_id}, destination {destination}, due {required_date}. The maximum total is EUR {budget} and minimum reliability is {reliability}. {route} Select by lowest emissions and then lowest cost before submitting.",
        "Which procurement plan should we use for {quantity} {unit} of {material_id} in {destination} by {required_date}? Enforce a EUR {budget} total budget and {reliability} minimum delivery reliability. {route} Carbon performance is more important than price. Use the tools to reach a final decision.",
        "Resolve the sourcing request for {material_id}: deliver {quantity} {unit} to {destination} by {required_date}, spending no more than EUR {budget} with reliability of at least {reliability}. {route} Prefer lower-carbon feasible plans, then cheaper ones, and finish through the tools.",
    ),
    "Spanish": (
        "Necesitamos {quantity} {unit} de {material_id} entregados en {destination} antes del {required_date}. El coste total no puede superar EUR {budget} y la fiabilidad de la entrega debe ser como mínimo del {reliability}. {route} Prioriza menos emisiones de carbono y después un menor coste. Usa las herramientas y finaliza con un plan o indicando que no hay una opción viable.",
        "Gestiona la compra de {quantity} {unit} de {material_id} para {destination}, con llegada máxima el {required_date}. Mantén el pedido completo por debajo de EUR {budget} y exige al menos un {reliability} de fiabilidad en la entrega. {route} Entre las opciones viables, valora primero el impacto de carbono y luego el precio. Completa la tarea mediante las herramientas.",
        "¿Puedes conseguir {material_id}? Hacen falta {quantity} {unit} en {destination} para el {required_date}, con un gasto total máximo de EUR {budget} y una fiabilidad de entrega mínima del {reliability}. {route} Minimiza primero las emisiones y después el coste, y presenta el resultado usando las herramientas.",
        "Solicitud de compras: {material_id}, {quantity} {unit}, destino {destination}, fecha límite {required_date}. Los límites obligatorios son EUR {budget} en total y un {reliability} de fiabilidad mínima de entrega. {route} Entre las alternativas válidas, favorece menos carbono y luego menor coste. Resuelve la solicitud con las herramientas disponibles.",
        "Para entregar en {destination} antes del {required_date}, adquiere {quantity} {unit} de {material_id}. No superes EUR {budget} y exige una fiabilidad de entrega de al menos {reliability}. {route} La reducción de carbono es prioritaria y el precio es secundario. Termina con la herramienta final adecuada.",
        "Necesito ayuda para tramitar un pedido de {quantity} {unit} de {material_id}. Debe llegar a {destination} como máximo el {required_date}, costar en total menos de EUR {budget} y ofrecer al menos un {reliability} de fiabilidad de entrega. {route} Prefiere la alternativa viable más ecológica y usa el precio como segundo criterio. Gestiona la solicitud con las herramientas.",
        "Encuentra una forma válida de suministrar {quantity} {unit} de {material_id} en {destination} antes del {required_date}. El presupuesto completo es de EUR {budget} y el mínimo de fiabilidad de entrega es {reliability}. {route} Tras cumplir las condiciones obligatorias, optimiza primero el carbono y luego el coste. Devuelve la decisión mediante una herramienta final.",
        "Estamos planificando un envío de {material_id}: se requieren {quantity} {unit} en {destination} para el {required_date}. Descarta cualquier opción que supere EUR {budget} en total o quede por debajo del {reliability} de fiabilidad de entrega. {route} Entre las opciones restantes, importan más las emisiones que el precio. Completa la selección con las herramientas.",
        "Consigue {quantity} {unit} de {material_id} para {destination}. Fecha límite: {required_date}; presupuesto total máximo: EUR {budget}; fiabilidad mínima de entrega: {reliability}. {route} Elige primero por menor carbono y después por menor coste, respetando esos límites, y presenta el resultado con las herramientas.",
        "¿Puedes comparar las opciones de compra para {quantity} {unit} de {material_id}, que deben estar en {destination} antes del {required_date}? No podemos gastar más de EUR {budget} en total ni aceptar menos del {reliability} de fiabilidad de entrega. {route} La sostenibilidad es la preferencia principal y el coste la segunda. Finaliza el proceso usando las herramientas.",
        "Prepara un plan de compra para {quantity} {unit} de {material_id} con entrega en {destination} antes del {required_date}. El total debe mantenerse dentro de EUR {budget} y la fiabilidad no puede ser inferior a {reliability}. {route} Ordena las opciones viables por huella de carbono y después por coste. Completa el proceso con las herramientas.",
        "Nuestro centro de {destination} necesita {quantity} {unit} de {material_id} como máximo el {required_date}. Aplica un límite total de EUR {budget} y una fiabilidad mínima de {reliability}. {route} Prefiere la opción válida con menos emisiones y usa el coste para desempatar. Presenta el resultado mediante las herramientas.",
        "Identifica una fuente válida para {quantity} {unit} de {material_id}, entregados en {destination} antes del {required_date}. Impón EUR {budget} como coste total máximo y {reliability} como mínimo de fiabilidad. {route} Optimiza primero el carbono y luego el precio, y cierra la solicitud con una herramienta final.",
        "Atiende esta necesidad de compra: {quantity} {unit} de {material_id} para {destination} antes del {required_date}. El pedido debe costar como máximo EUR {budget} y alcanzar al menos {reliability} de fiabilidad. {route} Entre las opciones conformes, elige primero por menos emisiones y después por menor coste. Finaliza con las herramientas.",
        "Selecciona proveedor y envío para {material_id}: {quantity} {unit} deben llegar a {destination} antes del {required_date}. El gasto total está limitado a EUR {budget} y la fiabilidad no puede bajar de {reliability}. {route} Usa el impacto de carbono como preferencia principal y el coste como secundaria. Finaliza con las herramientas.",
        "Tenemos una necesidad urgente de {quantity} {unit} de {material_id} en {destination} antes del {required_date}. No superes el presupuesto total de EUR {budget} y exige al menos {reliability} de fiabilidad. {route} Entre los planes viables, minimiza las emisiones antes que el coste. Registra la decisión con las herramientas.",
        "Revisa las posibilidades de compra de {material_id}, cantidad {quantity} {unit}, para {destination} antes del {required_date}. Excluye planes de más de EUR {budget} o con fiabilidad inferior a {reliability}. {route} Después favorece el transporte más ecológico y, en segundo lugar, el menor coste. Termina con las herramientas.",
        "Construye la mejor opción de compra válida para {quantity} {unit} de {material_id} con destino a {destination}. Debe llegar antes del {required_date}, ajustarse a EUR {budget} y ofrecer al menos {reliability} de fiabilidad. {route} Prefiere ahorrar carbono antes que dinero. Completa el flujo de herramientas.",
        "Determina cómo obtener {quantity} {unit} de {material_id} para {destination} antes del {required_date}. Considera EUR {budget} y {reliability} de fiabilidad como restricciones obligatorias. {route} Compara por emisiones y luego por coste total, y presenta el resultado con las herramientas.",
        "Solicitud para {destination}: suministra {quantity} {unit} de {material_id} antes del {required_date}. Ninguna opción puede superar EUR {budget} ni ofrecer menos de {reliability} de fiabilidad. {route} Elige el plan admisible más sostenible y usa después el coste. Acaba con la herramienta final correcta.",
        "Encuentra y presenta un plan viable para entregar {quantity} {unit} de {material_id} en {destination} no más tarde del {required_date}. El máximo es EUR {budget} y el umbral de fiabilidad {reliability}. {route} Favorece menos carbono y luego menor coste. Realiza toda la tarea con herramientas.",
        "Evalúa cómo debemos adquirir {material_id} para {destination}: necesitamos {quantity} {unit} antes del {required_date}. Mantén el coste total en EUR {budget} o menos y la fiabilidad en {reliability} o más. {route} La sostenibilidad prevalece sobre el precio entre alternativas válidas. Resuelve usando herramientas.",
        "Completa este pedido con las herramientas de compra: {quantity} {unit} de {material_id}, destino {destination}, fecha {required_date}. El máximo total es EUR {budget} y la fiabilidad mínima {reliability}. {route} Selecciona por menos emisiones y después por menor coste antes de presentar el resultado.",
        "¿Qué plan de compra debemos usar para {quantity} {unit} de {material_id} en {destination} antes del {required_date}? Aplica un presupuesto total de EUR {budget} y una fiabilidad mínima de {reliability}. {route} El rendimiento de carbono importa más que el precio. Usa las herramientas hasta tomar una decisión final.",
        "Resuelve la adquisición de {material_id}: entrega {quantity} {unit} en {destination} antes del {required_date}, gastando como máximo EUR {budget} con una fiabilidad mínima de {reliability}. {route} Prefiere planes viables con menos carbono y después los más baratos, y termina con las herramientas.",
    ),
    "German": (
        "Wir benötigen {quantity} {unit} {material_id} bis zum {required_date} in {destination}. Die Gesamtkosten dürfen EUR {budget} nicht überschreiten und die Lieferzuverlässigkeit muss mindestens {reliability} betragen. {route} Bevorzuge zuerst geringere CO₂-Emissionen und danach niedrigere Kosten. Nutze die Werkzeuge und schließe mit einem Beschaffungsplan oder einer Meldung über keine geeignete Option ab.",
        "Beschaffe bitte {quantity} {unit} {material_id} für {destination} mit Ankunft spätestens am {required_date}. Der gesamte Auftrag muss unter EUR {budget} bleiben und eine Lieferzuverlässigkeit von mindestens {reliability} haben. {route} Bewerte bei geeigneten Optionen zuerst den CO₂-Ausstoß und dann den Preis. Führe die Aufgabe mit den Werkzeugen aus.",
        "Kannst du {material_id} beschaffen? Benötigt werden {quantity} {unit} in {destination} bis {required_date}, bei Gesamtkosten von höchstens EUR {budget} und mindestens {reliability} Lieferzuverlässigkeit. {route} Minimiere zuerst Emissionen und danach Kosten und reiche das Ergebnis über die Werkzeuge ein.",
        "Beschaffungsanfrage: {material_id}, {quantity} {unit}, Zielort {destination}, Frist {required_date}. Verbindliche Grenzen sind EUR {budget} Gesamtkosten und mindestens {reliability} Lieferzuverlässigkeit. {route} Unter den gültigen Optionen sind weniger CO₂ und anschließend geringere Kosten zu bevorzugen. Löse die Anfrage mit den verfügbaren Werkzeugen.",
        "Für die Lieferung nach {destination} bis {required_date} werden {quantity} {unit} {material_id} benötigt. Überschreite EUR {budget} nicht und verlange mindestens {reliability} Lieferzuverlässigkeit. {route} CO₂-Reduktion hat Vorrang vor dem Preis. Beende die Aufgabe mit dem passenden Abschlusswerkzeug.",
        "Ich brauche Unterstützung bei einer Bestellung über {quantity} {unit} {material_id}. Sie muss spätestens am {required_date} in {destination} eintreffen, insgesamt unter EUR {budget} kosten und mindestens {reliability} Lieferzuverlässigkeit bieten. {route} Bevorzuge die umweltfreundlichste geeignete Wahl und nutze den Preis als zweites Kriterium. Bearbeite die Anfrage mit den Werkzeugen.",
        "Finde eine zulässige Möglichkeit, {quantity} {unit} {material_id} bis {required_date} nach {destination} zu liefern. Das Gesamtbudget beträgt EUR {budget}, die Mindestanforderung an die Lieferzuverlässigkeit {reliability}. {route} Nach Erfüllung aller Pflichtbedingungen optimiere zuerst CO₂ und dann Kosten. Gib die Entscheidung über ein Abschlusswerkzeug zurück.",
        "Wir planen eine Lieferung von {material_id}: {quantity} {unit} werden bis {required_date} in {destination} benötigt. Schließe alles aus, was insgesamt mehr als EUR {budget} kostet oder unter {reliability} Lieferzuverlässigkeit liegt. {route} Bei den übrigen Optionen sind geringere Emissionen wichtiger als ein niedrigerer Preis. Schließe die Auswahl mit den Werkzeugen ab.",
        "Beschaffe {quantity} {unit} {material_id} für {destination}. Frist: {required_date}; maximales Gesamtbudget: EUR {budget}; minimale Lieferzuverlässigkeit: {reliability}. {route} Wähle innerhalb dieser Grenzen zuerst nach dem geringsten CO₂-Ausstoß und danach nach den geringsten Kosten und reiche das Ergebnis über die Werkzeuge ein.",
        "Kannst du die Beschaffungsoptionen für {quantity} {unit} {material_id} vergleichen, die bis {required_date} in {destination} sein müssen? Insgesamt dürfen wir nicht mehr als EUR {budget} ausgeben und keine Lieferzuverlässigkeit unter {reliability} akzeptieren. {route} Nachhaltigkeit ist die wichtigste Präferenz, gefolgt von den Kosten. Beende den Ablauf mit den Werkzeugen.",
        "Erstelle einen Beschaffungsplan für {quantity} {unit} {material_id} mit Lieferung nach {destination} bis {required_date}. Die Gesamtkosten müssen innerhalb von EUR {budget} bleiben und die Lieferzuverlässigkeit darf {reliability} nicht unterschreiten. {route} Ordne geeignete Ergebnisse zuerst nach CO₂-Ausstoß und dann nach Kosten. Schließe den Vorgang mit den Werkzeugen ab.",
        "Unser Standort in {destination} benötigt spätestens am {required_date} {quantity} {unit} {material_id}. Es gilt eine Gesamtkostengrenze von EUR {budget} und eine Mindestzuverlässigkeit von {reliability}. {route} Bevorzuge die gültige Option mit den geringsten Emissionen und nutze die Kosten als zweites Kriterium. Reiche das Ergebnis über die Werkzeuge ein.",
        "Ermittle eine zulässige Bezugsquelle für {quantity} {unit} {material_id}, geliefert nach {destination} bis {required_date}. Setze EUR {budget} als maximale Gesamtkosten und {reliability} als Zuverlässigkeitsgrenze durch. {route} Optimiere zuerst CO₂ und dann den Preis und schließe die Anfrage mit einem Abschlusswerkzeug.",
        "Bearbeite diesen Einkaufsbedarf: {quantity} {unit} {material_id} für {destination} bis {required_date}. Die Bestellung darf höchstens EUR {budget} kosten und muss mindestens {reliability} Lieferzuverlässigkeit erreichen. {route} Wähle unter den zulässigen Optionen zuerst nach weniger Emissionen und dann nach niedrigeren Kosten. Nutze zum Abschluss die Werkzeuge.",
        "Wähle Lieferant und Versand für {material_id}: {quantity} {unit} müssen bis {required_date} in {destination} eintreffen. Die Gesamtausgaben sind auf EUR {budget} begrenzt, die Zuverlässigkeit muss mindestens {reliability} betragen. {route} Nutze den CO₂-Ausstoß als wichtigste und die Kosten als zweite Präferenz. Schließe mit den Werkzeugen ab.",
        "Wir benötigen dringend {quantity} {unit} {material_id} bis {required_date} in {destination}. Halte das Gesamtbudget von EUR {budget} ein und fordere mindestens {reliability} Lieferzuverlässigkeit. {route} Minimiere bei geeigneten Plänen Emissionen vor Kosten. Dokumentiere die Entscheidung mit den Werkzeugen.",
        "Prüfe Beschaffungsmöglichkeiten für {material_id}, Menge {quantity} {unit}, Lieferung nach {destination} bis {required_date}. Schließe Pläne über EUR {budget} oder unter {reliability} Zuverlässigkeit aus. {route} Bevorzuge anschließend umweltfreundlichere Lieferungen und danach niedrigere Kosten. Beende die Aufgabe mit den Werkzeugen.",
        "Erstelle die beste zulässige Einkaufslösung für {quantity} {unit} {material_id} nach {destination}. Sie muss bis {required_date} eintreffen, innerhalb von EUR {budget} bleiben und mindestens {reliability} Zuverlässigkeit bieten. {route} CO₂-Einsparungen sind wichtiger als Preiseinsparungen. Führe den Werkzeugablauf vollständig aus.",
        "Bestimme, wie {quantity} {unit} {material_id} bis {required_date} für {destination} beschafft werden können. Behandle EUR {budget} und {reliability} Lieferzuverlässigkeit als feste Grenzen. {route} Vergleiche geeignete Optionen zuerst nach Emissionen und dann nach Gesamtkosten und reiche das Ergebnis über die Werkzeuge ein.",
        "Einkaufsanfrage für {destination}: Liefere {quantity} {unit} {material_id} bis {required_date}. Keine Option darf insgesamt mehr als EUR {budget} kosten oder weniger als {reliability} Zuverlässigkeit bieten. {route} Wähle den nachhaltigsten zulässigen Plan und berücksichtige danach die Kosten. Beende mit dem richtigen Abschlusswerkzeug.",
        "Finde und übermittle einen praktikablen Plan, um {quantity} {unit} {material_id} spätestens am {required_date} nach {destination} zu liefern. Die Obergrenze beträgt EUR {budget}, die Zuverlässigkeitsschwelle {reliability}. {route} Bevorzuge weniger CO₂ und danach geringere Kosten. Führe die Aufgabe vollständig mit Werkzeugen aus.",
        "Bewerte, wie wir {material_id} für {destination} beschaffen sollten: Benötigt werden {quantity} {unit} bis {required_date}. Halte die Gesamtkosten bei höchstens EUR {budget} und die Zuverlässigkeit bei mindestens {reliability}. {route} Nachhaltigkeit steht bei gültigen Alternativen vor dem Preis. Löse die Aufgabe mit Werkzeugen.",
        "Erledige diese Bestellung mit den Beschaffungswerkzeugen: {quantity} {unit} {material_id}, Ziel {destination}, fällig am {required_date}. Das Gesamtmaximum beträgt EUR {budget}, die Mindestzuverlässigkeit {reliability}. {route} Wähle vor der Einreichung nach den geringsten Emissionen und dann den niedrigsten Kosten.",
        "Welchen Beschaffungsplan sollten wir für {quantity} {unit} {material_id} in {destination} bis {required_date} verwenden? Setze ein Gesamtbudget von EUR {budget} und mindestens {reliability} Lieferzuverlässigkeit durch. {route} Die CO₂-Bilanz ist wichtiger als der Preis. Nutze die Werkzeuge bis zur endgültigen Entscheidung.",
        "Löse die Beschaffung von {material_id}: Liefere {quantity} {unit} bis {required_date} nach {destination}, gib höchstens EUR {budget} aus und erreiche mindestens {reliability} Zuverlässigkeit. {route} Bevorzuge geeignete Pläne mit weniger CO₂ und danach günstigere und schließe mit den Werkzeugen ab.",
    ),
    "French": (
        "Nous avons besoin de {quantity} {unit} de {material_id} livrés à {destination} avant le {required_date}. Le coût total ne doit pas dépasser EUR {budget} et la fiabilité de livraison doit atteindre au moins {reliability}. {route} Privilégie d'abord la réduction des émissions de carbone, puis le coût. Utilise les outils et termine par un plan ou un constat d'absence d'option viable.",
        "Merci d'organiser l'achat de {quantity} {unit} de {material_id} pour {destination}, avec une arrivée au plus tard le {required_date}. La commande complète doit rester sous EUR {budget} et garantir au moins {reliability} de fiabilité de livraison. {route} Parmi les choix viables, évalue d'abord l'impact carbone, puis le prix. Effectue la tâche avec les outils.",
        "Peux-tu trouver {material_id} ? Il faut {quantity} {unit} à {destination} pour le {required_date}, avec un budget total maximal de EUR {budget} et une fiabilité de livraison minimale de {reliability}. {route} Réduis d'abord les émissions, puis le coût, et soumets le résultat avec les outils.",
        "Demande d'achat : {material_id}, {quantity} {unit}, destination {destination}, échéance {required_date}. Les limites obligatoires sont EUR {budget} au total et {reliability} de fiabilité minimale de livraison. {route} Parmi les options valides, favorise un carbone plus faible, puis un coût inférieur. Résous la demande avec les outils disponibles.",
        "Pour une livraison à {destination} avant le {required_date}, procure {quantity} {unit} de {material_id}. Ne dépasse pas EUR {budget} et exige une fiabilité de livraison d'au moins {reliability}. {route} La réduction du carbone est prioritaire et le prix secondaire. Termine avec l'outil final approprié.",
        "J'ai besoin d'aide pour passer une commande de {quantity} {unit} de {material_id}. Elle doit arriver à {destination} au plus tard le {required_date}, coûter moins de EUR {budget} au total et offrir au moins {reliability} de fiabilité de livraison. {route} Préfère le choix viable le plus écologique et utilise le prix comme second critère. Traite la demande avec les outils.",
        "Trouve une solution conforme pour livrer {quantity} {unit} de {material_id} à {destination} avant le {required_date}. Le budget global est de EUR {budget} et le seuil de fiabilité de livraison est {reliability}. {route} Une fois les contraintes obligatoires respectées, optimise d'abord le carbone, puis le coût. Renvoie la décision avec un outil final.",
        "Nous préparons une expédition de {material_id} : {quantity} {unit} sont nécessaires à {destination} pour le {required_date}. Écarte toute option dépassant EUR {budget} au total ou offrant moins de {reliability} de fiabilité de livraison. {route} Parmi les choix restants, les émissions comptent davantage que le prix. Termine la sélection avec les outils.",
        "Procure {quantity} {unit} de {material_id} pour {destination}. Échéance : {required_date} ; budget total maximal : EUR {budget} ; fiabilité minimale de livraison : {reliability}. {route} Dans ces limites, choisis d'abord le carbone le plus faible, puis le coût le plus bas, et soumets le résultat avec les outils.",
        "Peux-tu comparer les solutions d'achat pour {quantity} {unit} de {material_id}, attendues à {destination} avant le {required_date} ? Nous ne pouvons pas dépasser EUR {budget} au total ni accepter une fiabilité de livraison inférieure à {reliability}. {route} La durabilité est la préférence principale, suivie du coût. Termine le processus avec les outils.",
        "Prépare un plan d'achat pour {quantity} {unit} de {material_id}, à livrer à {destination} avant le {required_date}. Le total doit rester dans la limite de EUR {budget} et la fiabilité ne peut être inférieure à {reliability}. {route} Classe les résultats viables par empreinte carbone, puis par coût. Termine le processus avec les outils.",
        "Notre site de {destination} a besoin de {quantity} {unit} de {material_id} au plus tard le {required_date}. Applique un plafond total de EUR {budget} et une fiabilité minimale de {reliability}. {route} Préfère l'option valide la moins émettrice, puis départage par le coût. Soumets le résultat avec les outils.",
        "Identifie une source valide pour {quantity} {unit} de {material_id}, livrés à {destination} avant le {required_date}. Impose EUR {budget} comme coût global maximal et {reliability} comme seuil de fiabilité. {route} Optimise d'abord le carbone, puis le prix, et clôture la demande avec un outil final.",
        "Traite ce besoin d'achat : {quantity} {unit} de {material_id} pour {destination} avant le {required_date}. La commande doit coûter au maximum EUR {budget} et atteindre au moins {reliability} de fiabilité. {route} Parmi les options conformes, choisis d'abord les émissions les plus faibles, puis le coût. Termine avec les outils.",
        "Sélectionne un fournisseur et un transport pour {material_id} : {quantity} {unit} doivent arriver à {destination} avant le {required_date}. La dépense totale est limitée à EUR {budget} et la fiabilité doit atteindre {reliability}. {route} Utilise l'impact carbone comme préférence principale et le coût comme préférence secondaire. Finalise avec les outils.",
        "Nous avons un besoin urgent de {quantity} {unit} de {material_id} à {destination} avant le {required_date}. Respecte le budget total de EUR {budget} et exige au moins {reliability} de fiabilité. {route} Parmi les plans viables, minimise les émissions avant le coût. Enregistre la décision avec les outils.",
        "Examine les possibilités d'achat de {material_id}, quantité {quantity} {unit}, pour {destination} avant le {required_date}. Écarte les plans supérieurs à EUR {budget} ou inférieurs à {reliability} de fiabilité. {route} Favorise ensuite la livraison la plus écologique, puis le coût le plus faible. Termine avec les outils.",
        "Construis la meilleure option d'achat conforme pour {quantity} {unit} de {material_id} à destination de {destination}. Elle doit arriver avant le {required_date}, rester dans EUR {budget} et offrir au moins {reliability} de fiabilité. {route} Préfère les économies de carbone aux économies de prix. Termine le parcours d'outils.",
        "Détermine comment obtenir {quantity} {unit} de {material_id} pour {destination} avant le {required_date}. Considère EUR {budget} et {reliability} de fiabilité comme des contraintes strictes. {route} Compare les choix valides par émissions, puis par coût total, et soumets le résultat avec les outils.",
        "Demande d'achat pour {destination} : fournis {quantity} {unit} de {material_id} avant le {required_date}. Aucune option ne doit dépasser EUR {budget} ni offrir moins de {reliability} de fiabilité. {route} Choisis le plan admissible le plus durable, puis considère le coût. Termine avec le bon outil final.",
        "Trouve et soumets un plan réalisable pour livrer {quantity} {unit} de {material_id} à {destination} au plus tard le {required_date}. Le plafond est EUR {budget} et le seuil de fiabilité {reliability}. {route} Favorise un carbone plus faible, puis un coût inférieur. Effectue toute la tâche avec les outils.",
        "Évalue comment acheter {material_id} pour {destination} : il faut {quantity} {unit} avant le {required_date}. Maintiens le coût total à EUR {budget} ou moins et la fiabilité à {reliability} ou plus. {route} La durabilité prime sur le prix parmi les solutions valides. Résous et finalise avec les outils.",
        "Effectue cette commande avec les outils d'achat : {quantity} {unit} de {material_id}, destination {destination}, échéance {required_date}. Le maximum total est EUR {budget} et la fiabilité minimale {reliability}. {route} Sélectionne selon les émissions les plus faibles, puis le coût le plus bas avant de soumettre.",
        "Quel plan d'achat devons-nous utiliser pour {quantity} {unit} de {material_id} à {destination} avant le {required_date} ? Applique un budget total de EUR {budget} et une fiabilité minimale de {reliability}. {route} La performance carbone compte plus que le prix. Utilise les outils jusqu'à la décision finale.",
        "Résous l'approvisionnement de {material_id} : livre {quantity} {unit} à {destination} avant le {required_date}, sans dépasser EUR {budget} et avec au moins {reliability} de fiabilité. {route} Préfère les plans viables moins carbonés, puis les moins chers, et termine avec les outils.",
    ),
}


def format_prompt(scenario, language="English", template_index=0):
    """Render one scenario without changing its executable semantics."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    templates = PROMPT_TEMPLATES[language]
    if not 0 <= template_index < len(templates):
        raise ValueError(f"template_index must be between 0 and {len(templates) - 1}")
    route = ROUTES[language][scenario["task_type"]].format(**scenario)
    return templates[template_index].format(
        **scenario,
        budget=f"{scenario['maximum_total_cost']:.0f}",
        reliability=f"{scenario['minimum_delivery_reliability']:.0%}",
        route=route,
    )


def prompt_variants(
    scenario,
    languages=LANGUAGES,
    template_indices=range(1, TEMPLATES_PER_LANGUAGE + 1),
):
    """Yield prompt variants that retain the semantic scenario identifier."""
    template_indices = list(template_indices)
    if not template_indices or any(
        not 1 <= index <= TEMPLATES_PER_LANGUAGE for index in template_indices
    ):
        raise ValueError(
            f"template indices must be between 1 and {TEMPLATES_PER_LANGUAGE}"
        )
    for language in languages:
        for template_number in template_indices:
            template_index = template_number - 1
            variant = deepcopy(scenario)
            variant["language"] = language
            variant["prompt_variant"] = f"{language.lower()}_{template_number}"
            variant["user_request"] = format_prompt(
                variant, language, template_index
            )
            yield variant
