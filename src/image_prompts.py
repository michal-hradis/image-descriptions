# Dictionary of prompts for different processing modes
all_prompts =  {
    "orbis_01":
        {
            "keywords": (
                'Write list of keywords that describe the image. The keywords could describe for example the type '
                'of the image itself, objects, actions, location, names. Write only a list of the keywords  '
                'separated by commas.'
            ),
            "caption": (
                'Write five possible captions for the image which could for example be used in a book, magazine, '
                'webpage, newspaper or social media post. Write only a list of the captions with each caption on '
                'a separate text line.'
            ),
            "description": (
                'Describe the image in a few sentences. Describe the image itsef (photo, drawing, graphs), '
                'possibly it\'s style and historic period. Describe the image content, objects, actions, '
                'location, names.'
            ),
        },
    "aisee_01":
        {
            "person": (
                'Describe the person in the image using as a plain list of attributes. '
                'List the person\'s general appearance, age, ethnicity, hairstyle, body type, ... '
                'Provide just comma separated list of attributes. Do not write any additional text or comments.'
            ),
            "clothing": (
                'Describe each piece of clothing the person is wearing, all accessories, any carried items, '
                'and object the person is interacting with such as a phone, book, or computer, bicicles, '
                'suitcases, stroller and similar. Provide just s plain list of attributes such as: '
                'Provide just comma separated list of items. Do not write any additional text or comments.'
            ),
            "search_queries": (
                'Write a list of 10 search queries that a policeman could used to find this specific person '
                'from the image in a large database. Make the queries distinctive and specific for this person. '
                'Focus only on the person, clothing, accessories, and objects in the '
                'person\'s possession. Write only a list of the search queries separated by commas.'
                'Do not write any additional text or comments.'
            ),
        }
}